

funcs = ["whisper_full_n_segments", "whisper_full_get_segment_t0", "whisper_full_get_segment_t1",
"whisper_full_get_segment_text", "whisper_full_n_tokens", "whisper_full_get_token_text", "whisper_full_get_token_id",
"whisper_token_data whisper_full_get_token_data", "whisper_full_get_token_p", "whisper_print_system_info"]


for func in funcs:
    print(f"test.def(\"{func}\", &{func},\"{func}\");\n")