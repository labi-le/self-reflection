# syntax=docker/dockerfile:1

# Stage 1: build the Go parser binary (modernc sqlite is pure-Go, so CGO can stay off).
FROM golang:1.26-bookworm AS gobuild
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /out/parse ./cmd/parse

# Stage 2: build whisper.cpp's whisper-cli (voice transcription), statically linked.
FROM debian:bookworm-slim AS whisperbuild
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp /whisper \
    && cmake -S /whisper -B /whisper/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /whisper/build -j --target whisper-cli \
    && cp /whisper/build/bin/whisper-cli /usr/local/bin/whisper-cli

# Stage 3: runtime — parse binary + whisper-cli + ffmpeg.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=gobuild /out/parse /usr/local/bin/parse
COPY --from=whisperbuild /usr/local/bin/whisper-cli /usr/local/bin/whisper-cli
WORKDIR /export
ENTRYPOINT ["/usr/local/bin/parse"]
