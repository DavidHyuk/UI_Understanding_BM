#!/bin/bash
set -e

echo "Starting MediaPipe build process for aarch64 (Hybrid Strategy + Version Fix)..."

# 1. 저장소 클론
if [ ! -d "mediapipe_source" ]; then
    git clone https://github.com/google/mediapipe.git mediapipe_source
fi

cd mediapipe_source

# 2. Master 브랜치에서 최신 Dockerfile 확보
echo "Fetching Dockerfile from master..."
git reset --hard
git clean -fd
git checkout master
git pull origin master
cp Dockerfile.manylinux2014_aarch64rp4 /tmp/Dockerfile.saved

# 3. 안정적인 v0.10.14 버전으로 전환
echo "Checking out stable version v0.10.14..."
git reset --hard
git clean -fd
git checkout v0.10.14

# 4. Dockerfile 복원 및 패치
mv /tmp/Dockerfile.saved Dockerfile.manylinux2014_aarch64rp4
DOCKERFILE="Dockerfile.manylinux2014_aarch64rp4"

# Bazel 버전 수정 (v0.10.14는 Bazel 6.x 필요)
sed -i 's/7.4.1/6.1.1/g' $DOCKERFILE

# Protobuf 버전 수정
sed -i 's/v5.28.3/v28.3/g' $DOCKERFILE
sed -i 's/protoc-5.28.3/protoc-28.3/g' $DOCKERFILE
sed -i 's/curl -OL/curl -L -OL/g' $DOCKERFILE

# OpenCV 의존성 및 빌드 설정 수정 (v0.10.14 대응)
sed -i 's/yum install -y wget gcc-c++/yum install -y wget gcc-c++ libjpeg-turbo-devel libpng-devel libtiff-devel/g' $DOCKERFILE
sed -i 's/-DBUILD_LIST=imgproc,core//g' $DOCKERFILE
sed -i 's/srcs = \["lib64\/libopencv_imgproc.a", "lib64\/libopencv_core.a"\]/srcs = glob(["lib*\/libopencv_*.a"])/g' $DOCKERFILE
sed -i 's/includes = \["include\/opencv4\/"\]/includes = ["include\/opencv4\/", "include\/"]/g' $DOCKERFILE
sed -i 's/linkstatic = 1/linkstatic = 1, linkopts = ["-lpthread", "-ldl", "-lm"]/g' $DOCKERFILE

# 의존성 에러 방지
sed -i 's/RUN patch -p1 < mediapipe_python_build.diff/# RUN patch -p1 < mediapipe_python_build.diff/g' $DOCKERFILE
sed -i 's/pip install wheel/pip install "setuptools<70" wheel/g' $DOCKERFILE

# Python 버전 3.11로 변경
sed -i 's/Python-3.12.0/Python-3.11.10/g' $DOCKERFILE
sed -i 's/3.12.0/3.11.10/g' $DOCKERFILE
sed -i 's/cp312-cp312/cp311-cp311/g' $DOCKERFILE
sed -i 's/python3.12/python3.11/g' $DOCKERFILE

# ★ 버전 파싱 오류 수정 (2줄 삭제 및 대체)
# Dockerfile 구문 오류를 방지하기 위해 멀티라인 명령을 완전히 대체합니다.
sed -i '/RUN MP_VERSION_NUMBER/,+1c RUN sed -i "s/__version__ = '\''dev'\''/__version__ = '\''0.10.14'\''/g" setup.py' $DOCKERFILE

# Bazel 설정에 ENABLE_ODML_CONVERTER 추가
echo "build --define=ENABLE_ODML_CONVERTER=1" >> .bazelrc

# mediapipe/python/BUILD 패치 (하드코딩된 OpenCV 링크 옵션 제거 및 종속성 추가)
sed -i 's/"-lopencv_[^"]*",//g' mediapipe/python/BUILD
sed -i '/"@stblib\/\/:stb_image",/a \        "//third_party:opencv",' mediapipe/python/BUILD

# 5. Bazel 가시성 문제 해결
if [ -f "mediapipe/gpu/BUILD" ]; then
    if ! grep -q '"//visibility:public"' mediapipe/gpu/BUILD; then
        sed -i '/name = "_disable_gpu_flag",/a \    visibility = ["//visibility:public"],' mediapipe/gpu/BUILD
    fi
fi

# 6. Docker 빌드 실행
echo "Building Docker image (mp_aarch64)..."
sudo docker build -f $DOCKERFILE -t mp_aarch64 .

# 7. Wheel 결과물 추출 (이미 Docker 빌드 시점에 생성됨)
echo "Extracting the wheel from the container..."
mkdir -p dist
sudo docker run --rm -v $(pwd)/dist:/out mp_aarch64 /bin/bash -c "cp /wheelhouse/*.whl /out/"

# 8. 결과 확인
WHEEL_FILE=$(ls dist/*.whl 2>/dev/null | head -n 1)
if [ -f "$WHEEL_FILE" ]; then
    echo "-------------------------------------------"
    echo "Build successful: $WHEEL_FILE"
    echo "Copy this file to your DGX Spark and run:"
    echo "pip install $WHEEL_FILE"
    echo "-------------------------------------------"
else
    echo "Error: Wheel file not found. Check build logs."
    exit 1
fi

cd ..
