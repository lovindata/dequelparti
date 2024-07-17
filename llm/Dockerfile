FROM ollama/ollama:0.2.6 AS llm_builder
RUN apt update && \ 
    apt install -y curl
RUN bash -c "ollama serve > /dev/null 2>&1 &" && \
    curl --retry 10 --retry-connrefused --retry-delay 1 http://localhost:11434/ && \
    ollama pull gemma2:9b

FROM ollama/ollama:0.2.6
COPY --from=llm_builder /root/.ollama /root/.ollama
ENTRYPOINT [ "ollama", "serve" ]