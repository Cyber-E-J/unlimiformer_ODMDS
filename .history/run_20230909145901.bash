# Define the query as a variable
QUERY="What were the changes needed according to the group and the Professor's opinion on the use of English in the discussion about IBM computers and data collection participants?"

# Run the text extraction Python script
python src/extract.py --query "$QUERY"

# Read the extracted text into a variable
extracted_text=$(<extracted_text.txt)


python src/run_generation.py --model_type llama --model_name_or_path meta-llama/Llama-2-13b-chat-hf \
    --prefix "<<SYS>>\n You are a helpful assistant. Answer with detailed responses according to the entire instruction or question. \n<</SYS>>\n\n [INST] What were the changes needed according to the group and the Professor's opinion on the use of English in the discussion about IBM computers and data collection participants? " \
    --prompt extracted_text.txt \
    --suffix " [/INST]" --test_unlimiformer --fp16 --length 200 --layer_begin 16 \
    --index_devices 1 --datastore_device 1 