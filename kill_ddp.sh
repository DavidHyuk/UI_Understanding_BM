#!/bin/bash
# DDP 로 실행된 프로세스 강제 종료

set -e  # 에러 발생시 스크립트 중단

echo "=== DDP Process Killer ==="
echo "Searching and terminating DDP processes..."

# 함수 정의
cleanup_processes() {
    local signal=$1
    local signal_name=$2
    local wait_time=${3:-3}
    
    echo "Sending $signal_name signal to DDP processes..."
    
    # multiprocessing.spawn 프로세스 종료
    local spawn_pids=$(ps aux | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ ! -z "$spawn_pids" ]; then
        echo "Found multiprocessing.spawn processes: $spawn_pids"
        echo "$spawn_pids" | xargs -r kill $signal 2>/dev/null || true
    fi
    
    # torch.distributed.launch 프로세스 종료
    local launch_pids=$(ps aux | grep -E "(torch\.distributed\.launch|torchrun)" | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ ! -z "$launch_pids" ]; then
        echo "Found torch distributed launch processes: $launch_pids"
        echo "$launch_pids" | xargs -r kill $signal 2>/dev/null || true
    fi
    
    # Python 프로세스 중 DDP 관련 프로세스 종료
    pkill $signal -f "python.*run\.py" 2>/dev/null || true
    pkill $signal -f "python.*inference\.py" 2>/dev/null || true
    pkill $signal -f "torchrun" 2>/dev/null || true
    pkill $signal -f "torch\.distributed" 2>/dev/null || true
    
    # 잠시 대기
    if [ $wait_time -gt 0 ]; then
        echo "Waiting ${wait_time} seconds for processes to terminate..."
        sleep $wait_time
    fi
}

check_remaining_processes() {
    local remaining=$(ps aux | grep -E "(multiprocessing\.spawn|torch\.distributed|python.*run\.py|python.*inference\.py)" | grep -v grep | wc -l)
    echo "Remaining DDP processes: $remaining"
    return $remaining
}

# 1단계: SIGTERM으로 정상 종료 시도
cleanup_processes "-TERM" "SIGTERM" 3

# 2단계: 남은 프로세스 확인 후 SIGKILL로 강제 종료
if ! check_remaining_processes; then
    echo "Some processes still running. Using SIGKILL for force termination..."
    cleanup_processes "-KILL" "SIGKILL" 1
fi

# GPU 점유 프로세스 강제 종료
echo "Cleaning up GPU processes..."
for gpu_id in {0..7}; do
    if [ -e "/dev/nvidia$gpu_id" ]; then
        gpu_pids=$(lsof /dev/nvidia$gpu_id 2>/dev/null | grep python | awk '{print $2}' | sort -u 2>/dev/null || true)
        if [ ! -z "$gpu_pids" ]; then
            echo "Killing processes using GPU $gpu_id: $gpu_pids"
            echo "$gpu_pids" | xargs -r kill -9 2>/dev/null || true
        fi
    fi
done

# CUDA 프로세스 정리 (권한이 있는 경우만)
echo "Attempting GPU reset..."
if nvidia-smi --gpu-reset 2>/dev/null; then
    echo "GPU reset successful"
else
    echo "GPU reset failed (insufficient permissions - this is normal)"
fi

# 최종 확인
echo ""
echo "=== Final Status Check ==="
if check_remaining_processes; then
    echo "✅ All DDP processes have been terminated successfully!"
else
    echo "⚠️ Some processes may still be running. Manual intervention might be needed."
    echo "Remaining processes:"
    ps aux | grep -E "(multiprocessing\.spawn|torch\.distributed|python.*run\.py|python.*inference\.py)" | grep -v grep || echo "None found"
fi

echo "=== DDP Process Killer Completed ==="