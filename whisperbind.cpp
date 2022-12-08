/*
 *   Copyright (c) 2022 
 *   All rights reserved.
 */
#include "math.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "whisper.cpp/whisper.h"
#include "iostream"

#include "whisper.cpp/whisper.cpp"

namespace py = pybind11;
using namespace std;

double hypotenuse(double x, double y) {
    return sqrt(x * x + y * y);
}

PYBIND11_MODULE(whisperbind, test)
{
    test.def("hypotenuse", &hypotenuse, "Returns hypotenuse");

    // model enums
    py::enum_<e_model>(test, "e_model")
        .value("MODEL_UNKNOWN", e_model::MODEL_UNKNOWN)
        .value("MODEL_TINY",  e_model::MODEL_TINY)
        .value("MODEL_BASE",  e_model::MODEL_BASE)
        .value("MODEL_SMALL", e_model::MODEL_SMALL)
        .value("MODEL_MEDIUM",  e_model:: MODEL_MEDIUM)
        .value("MODEL_LARGE",  e_model::MODEL_LARGE)
        .export_values();

    // sampling strategy 
    py::enum_<whisper_sampling_strategy>(test, "whisper_sampling_strategy")
        .value("WHISPER_SAMPLING_GREEDY", whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY)
        .value("WHISPER_SAMPLING_BEAM_SEARCH", whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH)
        .export_values();

    //whisper_mel
    py::class_<whisper_mel>(test, "whisper_mel")
        .def_readwrite("n_len", &whisper_mel::n_len)
        .def_readwrite("n_mel", &whisper_mel::n_mel)
        .def_readwrite("data", &whisper_mel::data);


    //whisper filters
    py::class_<whisper_filters>(test, "whisper_filters")
        .def(py::init<>())
        .def_readwrite("n_mel", &whisper_filters::n_mel)
        .def_readwrite("n_fft", &whisper_filters::n_fft)
        .def("data", [](whisper_filters &self ) {
                                py::array out = py::cast(self.data);
                                return out;});

    // whisper vocab
    py::class_<whisper_vocab>(test, "whisper_vocab")
        .def(py::init<>())
        .def_readwrite("n_vocab", &whisper_vocab::n_vocab)
        .def_readwrite("token_to_id", &whisper_vocab::token_to_id)
        .def_readwrite("id_to_token", &whisper_vocab::id_to_token)

        .def_readwrite("token_eot", &whisper_vocab::token_eot)
        .def_readwrite("token_sot", &whisper_vocab::token_sot)

        .def_readwrite("token_prev", &whisper_vocab::token_prev)
        .def_readwrite("token_solm", &whisper_vocab::token_solm)
        .def_readwrite("token_not", &whisper_vocab::token_not)
        .def_readwrite("token_beg", &whisper_vocab::token_beg)
        .def("is_multilingual", &whisper_vocab::is_multilingual);
    
    //whisper segment
    py::class_<whisper_segment>(test, "whisper_segment")
        .def(py::init<>())
        .def_readwrite("t0", &whisper_segment::t0)
        .def_readwrite("t1", &whisper_segment::t1)
        .def_readwrite("tokens", &whisper_segment::tokens);


    // whisper hparams

    py::class_<whisper_hparams>(test, "whisper_hparams")
        .def(py::init<>())
        .def_readwrite("n_vocab", &whisper_hparams::n_vocab)
        .def_readwrite("n_audio_ctx", &whisper_hparams::n_audio_ctx)
        .def_readwrite("n_audio_state", &whisper_hparams::n_audio_state)
        .def_readwrite("n_audio_head", &whisper_hparams::n_audio_layer)
        
        .def_readwrite("n_audio_head", &whisper_hparams::n_audio_head)
        .def_readwrite("n_text_ctx", &whisper_hparams::n_text_ctx)
        .def_readwrite("n_text_state", &whisper_hparams::n_text_head)
        .def_readwrite("n_text_layer", &whisper_hparams::n_text_layer)

        .def_readwrite("n_mels", &whisper_hparams::n_mels)
        .def_readwrite("f16", &whisper_hparams::f16);

//     struct whisper_context {
//     int64_t t_load_us   = 0;
//     int64_t t_mel_us    = 0;
//     int64_t t_sample_us = 0;
//     int64_t t_encode_us = 0;
//     int64_t t_decode_us = 0;
//     int64_t t_start_us  = 0;

//     std::vector<uint8_t> * buf_model; // the model buffer is read-only and can be shared between processors
//     std::vector<uint8_t>   buf_memory;
//     std::vector<uint8_t>   buf_compute;
//     std::vector<uint8_t>   buf_compute_layer;

//     whisper_model model;
//     whisper_vocab vocab;

//     whisper_mel mel;

//     std::vector<float> probs;
//     std::vector<float> logits;

//     std::vector<whisper_segment> result_all;

//     std::vector<whisper_token> prompt_past;

//     // [EXPERIMENTAL] token-level timestamps data
//     int64_t t_beg;
//     int64_t t_last;
//     whisper_token tid_last;
//     std::vector<float> energy; // PCM signal energy
// };
    //Whisper Context
    py::class_<whisper_context>(test, "whisper_context")
        .def(py::init<>())
        .def_readwrite("t_load_us", &whisper_context::t_load_us)
        .def_readwrite("t_mel_us", &whisper_context::t_mel_us)
        .def_readwrite("t_sample_us", &whisper_context::t_sample_us)

        .def_readwrite("t_encode_us", &whisper_context::t_encode_us)
        .def_readwrite("t_decode_us", &whisper_context::t_decode_us)
        .def_readwrite("t_start_us", &whisper_context::t_start_us)

        .def("buf_memory", [](whisper_context &self ) {
                                py::array out = py::cast(self.buf_memory);
                                return out;})
        .def("buf_compute", [](whisper_context &self ) {
                                py::array out = py::cast(self.buf_compute);
                                return out;})
        
        .def("buf_compute_layer", [](whisper_context &self ) {
                                py::array out = py::cast(self.buf_compute);
                                return out;})

        .def_readwrite("model", &whisper_context::model)
        .def_readwrite("vocab", &whisper_context::vocab)
        .def_readwrite("mel", &whisper_context::mel)

        .def("probs", [](whisper_context &self ) {
                                py::array out = py::cast(self.probs);
                                return out;})
        .def("logits", [](whisper_context &self ) {
                                py::array out = py::cast(self.logits);
                                return out;})

        .def_readwrite("result_all", &whisper_context::result_all)
        .def_readwrite("prompt_past", &whisper_context::prompt_past)
        .def_readwrite("t_beg", &whisper_context::t_beg)
        .def_readwrite("t_last", &whisper_context::t_beg)
        .def("energy", [](whisper_context &self ) {
                                py::array out = py::cast(self.energy);
                                return out;});
        



    // Whisper Token Data
    py::class_<whisper_token_data>(test, "whisper_token_data") 
        .def_readwrite("id", &whisper_token_data::id)
        .def_readwrite("tid", &whisper_token_data::tid)
        .def_readwrite("p", &whisper_token_data::p)
        .def_readwrite("pt", &whisper_token_data::pt)
        .def_readwrite("ptsum", &whisper_token_data::ptsum)
        .def_readwrite("t0", &whisper_token_data::t0)
        .def_readwrite("t1", &whisper_token_data::t1);
    

    // Whisper Struct
    py::class_<whisper_full_params>(test, "whisper_full_params")
        .def(py::init<>())
        .def_readwrite("n_threads", &whisper_full_params::n_threads)
        .def_readwrite("n_max_text", &whisper_full_params::n_max_text_ctx)
        .def_readwrite("offset_ms", &whisper_full_params::offset_ms)

        .def_readwrite("translate", &whisper_full_params::translate)
        .def_readwrite("no_context", &whisper_full_params::no_context)
        .def_readwrite("print_special_tokens", &whisper_full_params::print_special_tokens)

        .def_readwrite("print_progress", &whisper_full_params::print_progress)
        .def_readwrite("print_realtime", &whisper_full_params::print_realtime)
        .def_readwrite("print_timestamps", &whisper_full_params::print_timestamps)

        .def_readwrite("token_timestamps", &whisper_full_params::token_timestamps)
        .def_readwrite("thold_pt", &whisper_full_params::thold_pt)
        .def_readwrite("thold_ptsum", &whisper_full_params::thold_ptsum)

        .def_readwrite("max_len", &whisper_full_params::max_len)
        .def_readwrite("speed_up", &whisper_full_params::speed_up)
        .def_readwrite("language", &whisper_full_params::language)
        .def_readwrite("thold_ptsum", &whisper_full_params::thold_ptsum);

    // whisper model init
    test.def("whisper_init", &whisper_init, "Initializes whisper model");
    // whisper model load
    test.def("whisper_model_load", &whisper_model_load, "Loads the whisper model");

    // // whisper free
    test.def("whisper_free", &whisper_free, "Free the whisper struct");

    // // whisper_pcm_to_mel
    test.def("whisper_pcm_to_mel", &whisper_pcm_to_mel, "whisper pcm to mel func");

    // // whisper_pcm_to_mel_phase_vocoder
    test.def("whisper_pcm_to_mel_phase_vocoder", &whisper_pcm_to_mel_phase_vocoder, "whisper_pcm_to_mel_phase_vocoder");

    // // whisper set mel
    test.def("whisper_set_mel", &whisper_pcm_to_mel, "whisper_set_mel");

    // // whsiper_encode//
    test.def("whisper_encode", [](struct whisper_context * ctx, int offset, int n_threads) {
        return whisper_encode(ctx, offset, n_threads);
    }, "whisper_encode");

    // whisper_decode
    test.def("whisper_decode", [](struct whisper_context * ctx, const whisper_token * tokens, int n_tokens, int n_past, int n_threads){
        return whisper_decode(ctx, tokens, n_tokens, n_past, n_threads);
    }, "whisper_decode");

    //whisper_lang_id
    test.def("whisper_lang_id", &whisper_lang_id, "whisper_lang_id");

    // //whisper_sample_best
    test.def("whisper_sample_best", [](struct whisper_context * ctx){
        return whisper_sample_best(ctx);
    }, "whisper_sample_best");

    // //whisper_sample_timestamp
    test.def("whisper_sample_timestamp", [](struct whisper_context * ctx){
        return whisper_sample_timestamp(ctx);
    } , "whisper_sample_timestamp");

    test.def("whisper_n_len", &whisper_n_len, "whisper_n_len");

    test.def("whisper_n_vocab", &whisper_n_vocab, "whisper_n_vocab");

    test.def("whisper_n_text_ctx", &whisper_n_text_ctx, "whisper_n_text_ctx");

    test.def("whisper_is_multilingual", &whisper_is_multilingual, "whisper_is_multilingual");

    test.def("whisper_get_probs", [] (struct whisper_context * ctx) {
        py::array out = py::cast(whisper_get_probs(ctx)); return out;}, "whisper_get_probs");

    test.def("whisper_token_to_str", &whisper_token_to_str, "whisper_token_to_str");   

    test.def("whisper_token_eot", &whisper_token_eot, "whisper_token_eot");  

    test.def("whisper_token_sot", &whisper_token_sot, "whisper_token_sot");
    
    test.def("whisper_token_prev", &whisper_token_prev, "whisper_token_prev"); 

    test.def("whisper_token_solm", &whisper_token_solm, "whisper_token_solm"); 

    test.def("whisper_token_not", &whisper_token_not, "whisper_token_not"); 

    test.def("whisper_token_beg", &whisper_token_beg, "whisper_token_beg"); 

    test.def("whisper_token_translate", &whisper_token_translate, "whisper_token_translate"); 

    test.def("whisper_token_transcribe", &whisper_token_transcribe, "whisper_token_transcribe"); 

    test.def("whisper_print_timings", &whisper_print_timings, "whisper_print_timings");

    test.def("whisper_full_default_params", &whisper_full_default_params, "whisper_full_default_params");

    test.def("whisper_wrap_segment", &whisper_wrap_segment, "whisper_wrap_segment");

    test.def("whisper_full", [](
        struct whisper_context * ctx,
        struct whisper_full_params params,
        py::array_t<float> &samples,
        int n_samples) { 
            int response;
            py::gil_scoped_release release;
            // int i;
            // cout << "\nIs this working???????\n";
            py::buffer_info sample_buff = samples.request(); 
            float* ptr1 = (float*) sample_buff.ptr;

            // for(i = 0; i < 30; i++) {
            //     cout << ptr1[i] << i << "\n";
            // }
            response = whisper_full(ctx,params,*ptr1,n_samples);
            py::gil_scoped_acquire acquire;
            return response;}, "whisper_full");

    test.def("whisper_full_parallel", &whisper_full_parallel, "whisper_full_parallel");

    test.def("whisper_full_n_segments", &whisper_full_n_segments,"whisper_full_n_segments");

    test.def("whisper_full_get_segment_t0", &whisper_full_get_segment_t0,"whisper_full_get_segment_t0");

    test.def("whisper_full_get_segment_t1", &whisper_full_get_segment_t1,"whisper_full_get_segment_t1");

    test.def("whisper_full_get_segment_text", &whisper_full_get_segment_text,"whisper_full_get_segment_text");

    test.def("whisper_full_n_tokens", &whisper_full_n_tokens,"whisper_full_n_tokens");

    test.def("whisper_full_get_token_text", &whisper_full_get_token_text,"whisper_full_get_token_text");

    test.def("whisper_full_get_token_id", &whisper_full_get_token_id,"whisper_full_get_token_id");

    test.def("whisper_full_get_token_data", &whisper_full_get_token_data,"whisper_full_get_token_data");

    test.def("whisper_full_get_token_p", &whisper_full_get_token_p,"whisper_full_get_token_p");

    test.def("whisper_print_system_info", &whisper_print_system_info,"whisper_print_system_info");


}