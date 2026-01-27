vllm bench serve \
 --backend vllm \
 --model /home/y00889327/y00889327_dsv3_w8a8_mtp/ \
 --endpoint /v1/completions \
 --dataset-name random \
 --random-input-len 2 \
 --random-output-len 100 \
 --max-concurrency 320 \
 --num-prompts 500 \
 --port 8006
