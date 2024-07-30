FROM scratch
ARG TARGETOS
ARG TARGETARCH
COPY .dist/gguf-parser-${TARGETOS}-${TARGETARCH} /bin/gguf-parser
ENTRYPOINT ["/bin/gguf-parser"]
