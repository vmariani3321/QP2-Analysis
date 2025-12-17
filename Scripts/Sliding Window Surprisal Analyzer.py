#########################################
"""
VINCENT MARIANI
12 DECEMBER 2025

Analysis script for QP 2; 
Takes an XML corpus (in this case the BNC1994) and outputs CSV files with surprisal calculations and NLP characteristics.
"""

#############################
# IMPORTS
#############################

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import torch
from tqdm import tqdm
import pandas as pd
import math
import collections
import os
import xml.etree.ElementTree as ET
import gc
from spacy.tokens import Doc
import time
from itertools import batched

############################
# CONFIG
############################

CSV_BATCH = 999999999 # The number of sentences to process before writing to the CSV file. Set to absurdly high number to process entire file at once

WINDOW_LEN = 4 # The length of the sliding window 
WINDOW_BATCH = 32 # The number of sliding windows to process at once
TOKEN_WARNING = 120000 # The number of tokens in a window to throw a warning

SPACY_BATCH = 64

OVERWRITE = 0

INPUT_DIR = "D:/BNC Full Data/BNCFiles/Full BNC1994/download/Texts" # The directory of the input XML files
OUTPUT_DIR = "D:/BNC Full Data/12-17 Run/CSV" # The directory of the output CSV files

SPACY_MOD = "en_core_web_trf" # The SpaCy model to use
TRANSFORMER_MOD = "meta-llama/Llama-3.2-1B" # The transformer model to use

############################
# HELPERS
############################

def timer(start_time, end_time):
    """
    Times a function. Mostly used for debugging and tuning.
    
    For example:

    start_time = time.time()
    {some function}
    end_time = time.time()

    timer(start_time, end_time)

    """
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    hhmmss_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"Total processing time: {hhmmss_duration}.")

####

def empty_gpu_cache():
    """
    A platform agnostic function to clear the GPU cache.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

####

def get_filepaths(inputDir):
    """
    Gathers sorted list of XMLs from a directory (recursive).
    """

    filepaths = []
    if not os.path.isdir(inputDir):
        print("Directory doesn't exist.")
        return [] # Exits if directory does not exist.
    
    for root, dirs, files in os.walk(inputDir): # Walks recursively through file tree
        files.sort()
        dirs.sort()
        for filename in files:
            if filename.endswith('.xml'): # Selects only XMLs
                filepaths.append(os.path.join(root, filename)) # Creates list of filepaths

    if not filepaths:
        print("No XMLs found")
        return [] # Exits if no XMLs exist
    
    return filepaths

####

def initialize_models(spacy_model, hf_model):
    """
    Activates NLP and LLM models.
    """
    
    
    # Prepare SpaCy #

    if torch.cuda.is_available():
        spacy.require_gpu()
        spacy_device = "GPU"
    else:
        spacy_device = "CPU"

    nlp = spacy.load(spacy_model)

    
    # Prepare HuggingFace Transformer Model (using Accelerator) #

    accelerator = Accelerator() # Memory management tool

    tokenizer = AutoTokenizer.from_pretrained(hf_model,use_fast=True) # Creates tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained( # Creates PyTorch model and pushes to GPU
        hf_model, device_map="auto", torch_dtype=torch.float16
    )
    model.eval() # Sets model to evaluation mode
    model = accelerator.prepare(model) # Stability wrapper; handles data types and memory allocation

    print(f"Transformer initialized on device {accelerator.device}.\nSpaCy model initialized on {spacy_device}.")

    return nlp, accelerator, tokenizer, model
    
####
    
def compute_iou(a_start, a_end, b_start, b_end):
    """
    Intersection over Union
    """
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0: return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0

####

def get_window(sentence_tuples, batch_num):
    """
    Generates context windows for analysis. 

    Params: 
        - sentence_tuples: The list of inputs to turn into windows. 
        - batch_num: the current CSV writing batch number
    """
    slice_start = batch_num * CSV_BATCH
    slice_end = min((batch_num + 1) * CSV_BATCH, len(sentence_tuples))

    for target_sentence in range(slice_start, slice_end):
        window_end = target_sentence + 1
        window_start = max(0, window_end - WINDOW_LEN)

        yield sentence_tuples[window_start:window_end]

############################
# DATA PROCESSING
############################
def XML_tupler(filepath):
    """
    Parses one XML into a list of (text, context) tuples.
    """

    sentence_tuples = [] # Creates holding list for tuples
    base_filename = os.path.basename(filepath) # Gathers filename w/out path
    filename_no_ext = os.path.splitext(base_filename)[0] # Gathers filename w/out .xml extension
    sentence_counter = 0 # For forming sentence ID number

    try:
        tree = ET.parse(str(filepath))
        root = tree.getroot() # Finds root of XML tree
    except ET.ParseError as e: # Sanity check for invalid XML
        print(f" Parse error {e}. Skipping file.")
        return []
    
    if root.find(".//wtext") is not None:
        modality = "written"
    elif root.findall (".//stext"):
        modality = "spoken"
    else:
        modality = "unknown"

    for sentence_tag in root.findall(".//s"): # Extracts sentences
        words = [
            child.text.strip() # Removes extra spaces
            for child in sentence_tag
            if child.tag in ['w', 'c'] and child.text is not None # Removes empty words
        ]
        
        if words:
            sentence_text = ' '.join(words).strip() # Joins words into a sentence with only one space between words and removes empty sentences
            sentence_counter += 1

            s_n = sentence_tag.get('n') # Get sentence number from XML

            if s_n:
                if s_n.isdigit():
                    sent_num = f"{int(s_n):04d}"
                else:
                    sent_num = s_n
            else: 
                sent_num = "xxxx"

            BNCID = f"{filename_no_ext}_{sent_num}"  # BNCID = BNC ID number, used for citations
            consecutive_ID = f"{filename_no_ext}_{sentence_counter:04d}" # Used in case BNC ID is not proper, mainly for sorting in R

            metadata = {
                "BNC_ID" : BNCID,
                "consecutive_ID" : consecutive_ID,
                "filename" : base_filename,
                "modality" : modality
            }
            sentence_tuples.append((sentence_text, metadata))

    return sentence_tuples


####

def surprisal_calc(sentence_tuples, tokenizer, model, accelerator, batch_num):
    """
    Uses sliding window approach to get surprisal values.

    * Requires from itertools import batched

    Params: 
        - sentence_tuples: The list of inputs to turn into windows
        - batch_num: The current CSV writing batch number (automatically fills)

    """
    final_results = []

    window_generator = get_window(sentence_tuples, batch_num)

    current_batch_size = min((batch_num + 1) * CSV_BATCH, len(sentence_tuples)) - (batch_num * CSV_BATCH) 
    total_windows = math.ceil(current_batch_size/WINDOW_BATCH)

    for batch_idx, batch in tqdm(enumerate(batched(window_generator, n = WINDOW_BATCH)), total=total_windows, desc="Processing Batch of Windows", position=2, leave=False):

        window_strings = []
        target_len = []
        target_info = []

        for window in batch: # window == [(Text, Context), (Text, Context), ...]
            window_text = [text[0] for text in window]

            window_strings.append(tokenizer.bos_token + " ".join(window_text))

            # Keep track of actual surprisal target: 

            target_text = window_text[-1]
            target_metadata = window[-1][1]
            target_info.append((target_text, target_metadata))

            # Target length for slicing: 
            target_prefix = " " if len(window_text) > 1 else ""
            t_len = len(tokenizer(target_prefix + target_text, add_special_tokens = False)['input_ids'])
            target_len.append(t_len)

        inputs = tokenizer(window_strings, return_tensors = 'pt', padding=True).to(accelerator.device)

        input_len = inputs['input_ids'].shape[1]
        max_len = tokenizer.model_max_length

        if input_len >= TOKEN_WARNING:
            print(f"WARN: {input_len} tokens in tensor; model limited to {max_len}. Exceeding this limit will crash the script.")

        with torch.no_grad():
            outputs = model(**inputs)

            log_probs = torch.log_softmax(outputs.logits, -1)
            gold_tokens = inputs['input_ids'][:,1:]
            gold_logprobs = log_probs[:, :-1, :]

            surprisal = -(torch.gather(gold_logprobs, dim = 2, index = gold_tokens.unsqueeze(-1)).squeeze(-1).to(torch.float16).cpu())

        del outputs, log_probs, gold_logprobs, gold_tokens, 

        for j, t_len in enumerate(target_len):
            real_len = inputs.attention_mask[j].sum().item()-1 # Removes padding

            # Slice the tokenized length 
            start_pos = real_len - t_len
            sentence_surprisal_slice = surprisal[j, start_pos:real_len]
            sentence_surprisal_list = sentence_surprisal_slice.tolist()

            original_text, original_metadata = target_info[j]

            current_metadata = original_metadata.copy()
            current_metadata['alignment_text'] = original_text


            final_results.append((original_text, current_metadata, sentence_surprisal_list))

            del sentence_surprisal_slice

        # del inputs

        # if batch_idx % 10 == 0:
        #     empty_gpu_cache()
        #     gc.collect()

    return final_results

def spacy_streamer(sentence_tuples, nlp):
    """
    Processes sentences into SpaCy docs but passes them on one at a time. 
    """

    if not Doc.has_extension("sentence_metadata"):
        Doc.set_extension("sentence_metadata", default = None) # Saves sentence metadata into a doc entry

    if not Doc.has_extension("sentence_surprisals"):
        Doc.set_extension("sentence_surprisals", default = None) # Same, but for surprisals

    compressed_tuples = [] # SpaCy can't handle triple tuples; converts them into doubles
    for item in sentence_tuples:
        text = item[0]
        context = (item[1], item[2])
        compressed_tuples.append((text, context))

    doc_pipe = nlp.pipe(compressed_tuples, as_tuples = True, batch_size = SPACY_BATCH)

    for doc, context in doc_pipe:
        metadata, surprisals = context
        doc._.sentence_metadata = metadata
        doc._.sentence_surprisals = surprisals

        yield doc    


def alignment(doc, tokenizer):
    """
    Align LLaMa tokens to SpaCy tokens
    """

    alignment_text = doc._.sentence_metadata.get('alignment_text', doc.text)

    shift_amount = len(alignment_text) - len(doc.text)

    offsets = tokenizer(alignment_text, add_special_tokens = False, return_offsets_mapping = True)['offset_mapping'] # Gather start and end characters for each token, stripping special tokens

    spacy_spans = [(tok.idx, tok.idx + len(tok), tok) for tok in doc] # Does the same for spacy; outputs (start, end, token)

    aligned, spacy_pointer = [], 0
    for llama_start, llama_end in offsets: # Loop through every LLaMa token

        adj_start = llama_start - shift_amount
        adj_end = llama_end - shift_amount

        if adj_end <= 0:
            aligned.append(None)
            continue


        best_tok, best_iou = None, 0.0

        while spacy_pointer < len(spacy_spans) and spacy_spans[spacy_pointer][1] <= llama_start: # Prevents starting from the beginning multiple times
            spacy_pointer += 1
        for i in range(spacy_pointer, min(len(spacy_spans), spacy_pointer + 5)): # Look 4 tokens ahead of current
            spacy_start, spacy_end, spacy_tok = spacy_spans[i]
            iou = compute_iou(adj_start, adj_end, spacy_start, spacy_end) # Calculates overlap between LLaMa and SpaCy
            if iou > best_iou: # Pick the best match
                best_iou, best_tok = iou, spacy_tok
        aligned.append(best_tok)
    return aligned

def tokenize_surprisal(doc, aligned):
    """
    Assigns surprisal values to each word
    """
    surprisals = doc._.sentence_surprisals # Takes sentence surprisal lists from doc
    mean_token_surprisals = collections.defaultdict(float)
    token_piece_counts = collections.defaultdict(int)
    for k, spacy_tok in enumerate(aligned): # For each aligned token
        if spacy_tok is not None and k < len(surprisals): # If a token exists and there are surprisals available
            mean_token_surprisals[spacy_tok.i] += surprisals[k] # Take the sum of the surprisals for that token
            token_piece_counts[spacy_tok.i] += 1 # Count the number of subwords in that token

    for idx in mean_token_surprisals: # For each token in the sentence
        mean_token_surprisals[idx] /= token_piece_counts[idx] # Take the average surprisal of its subtokens

    return mean_token_surprisals

### Collect data from the sentences

#!# Old extract_spacy_data

def generate_rows(doc, token_surprisals):
    """
    Generates CSV rows for each word, filtering metrics and word-level data, mean surprisals over entire NPs, etc.
    """

    #Initialize Counts
    verb_count = 0
    aux_count = 0
    subject_count = 0
    dir_obj_count = 0
    ind_obj_count = 0
    oth_obj_count = 0
    commas = 0
    sub_conj_count = 0
    coord_conj_count = 0
    relative_clause_count = 0
    adv_clause_count = 0
    clausal_comp_count = 0
    prep_phrase_count = 0

    # Loop over all tokens
    for token in doc: 
        # POS based counts
        if token.pos_ == "VERB":
            verb_count += 1
        elif token.pos_ == "AUX":
            aux_count += 1
        elif token.pos_ == "SCONJ":
            sub_conj_count += 1
        elif token.pos_ == "CCONJ":
            coord_conj_count += 1

        # DEP based counts
        dep = token.dep_
        if dep == "nsubj" and token.pos_ in ("NOUN", "PRON", "PROPN"):
            subject_count += 1
        elif dep == "dobj":
            dir_obj_count += 1
        elif dep in ("iobj", "dative"):
            ind_obj_count += 1
        # elif dep == "obj": # Doesn't appear to be used by spaCy
        #     oth_obj_count += 1
        elif dep == "relcl":
            relative_clause_count += 1
        elif dep == "advcl":
            adv_clause_count += 1
        elif dep == "ccomp":
            clausal_comp_count += 1
        elif dep == 'pobj':
            prep_phrase_count += 1

        # Text based counts
        if token.text == ",":
            commas += 1

    # Count derived values
    total_obj_count = dir_obj_count + ind_obj_count + oth_obj_count
    transitive = dir_obj_count > 0

    sentence_metadata = {
        "bnc_id" : doc._.sentence_metadata["BNC_ID"],
        "consecutive_id ": doc._.sentence_metadata["consecutive_ID"],
        "filename": doc._.sentence_metadata["filename"],
        "modality": doc._.sentence_metadata["modality"],
        "s_text": doc.text,
        "s_verb_cnt": verb_count,
        "s_aux_cnt": aux_count,
        "s_aux_cnt": subject_count,
        's_tot_obj_cnt': total_obj_count,
        's_dir_obj_cnt': dir_obj_count,
        's_ind_obj_cnt': ind_obj_count,
        "s_trans": transitive,
        "s_comma_cnt": commas,
        "s_sub_conj_cnt": sub_conj_count,
        's_coord_conj_cnt': coord_conj_count,
        "s_rel_clause_cnt": relative_clause_count,
        "s_adv_clause_cnt": adv_clause_count,
        "clausal_comp_cnt": clausal_comp_count,
        "s_pp_cnt": prep_phrase_count
    }


    token_rows = []

    # Create a template row for each token
    for token in doc:
        individual_surprisal = token_surprisals.get(token.i)
        base_row = {
            **sentence_metadata,
            'w_idx' : token.i,
            'w_tok' : token.text,
            'phr_tok' : token.text,
            'surprisal' : individual_surprisal,#Gets overrwritten by NPs
            'w_surp' : individual_surprisal, 
            'w_pos' : token.pos_,
            'w_dep' : token.dep_,
            'is_noun' : False,
            'is_bare_np' : None,
            'np_struct' : None, 
            'head_pos' : None, 'det_pos' : None,
            'head_dep' : None, 'det_dep' : None,
            'head_text' : None, 'det_text' : None,
            'np_sum_surp' : None, 'np_mean_surp' : None, 
            'argPos' : None, 'np_num' : None, 'definiteness' : None,
        }
        token_rows.append(base_row)

    # Process noun chunks and UPDATE existing rows, to prevent duplication.
    for np in doc.noun_chunks:
        head = np.root
        det = next((tok for tok in np if tok.dep_ in ("det", "poss")), None)

        # For creating list of elements in NP:
        pos_list = [tok.pos_ for tok in np]
        np_structure_string = " + ".join(pos_list)

        argument = "non-arg"
        if head.dep_ == "nsubj": argument = "sbj"
        elif head.dep_ == "obj": argument = "oth_object"
        elif head.dep_ == "dobj": argument = "obj"
        elif head.dep_ == "iobj": argument = "ind_object"
        elif head.dep_ == "pobj": argument = "prep_object"

        number = "unmarked"
        if "Number=Sing" in str(head.morph): number = "sing"
        elif "Number=Plur" in str(head.morph): number = "plur"

        definiteness = "unmarked"
        has_poss = any(tok.dep_ == 'poss' for tok in np)

        if has_poss:
            definiteness = "def"
        elif det:
            if "Definite=Def" in str(det.morph) or "Poss=Yes" in str(det.morph): definiteness = "def"
            if "Definite=Ind" in str(det.morph): definiteness = "indef"
        elif head.pos_ in ("PROPN", "PRON"): definiteness = "def"
        elif head.pos_ == "NOUN": definiteness = "indef"

# Find out how to get Head and Det surprisals
        np_surprisals = [token_surprisals.get(tok.i) for tok in np if token_surprisals.get(tok.i) is not None]
        sum_s = sum(np_surprisals) if np_surprisals else None
        mean_s = sum_s / len(np_surprisals) if np_surprisals else None
        # TRY THESE LATER
        # head_s = [token_surprisals.get(head.i) for head in np if token_surprisals.get(head.i) is not None]
        # det_s = [token_surprisals.get(det.i) for det in np if token_surprisals.get(det.i) is not None]

        np_data = {

            'phr_tok' : np.text,
            'surprisal' : mean_s,
            'is_noun' : True,
            'is_bare_np' : False if det else True,
            "np_struct" : np_structure_string,
            "is_np_head" : False,
            'head_pos' : head.pos_, 'det_pos' : det.pos_ if det else None,
            'head_dep' : head.dep_, 'det_dep' : det.dep_ if det else None,
            'head_text' : head.text, 'det_text' : det.text if det else None,
            'np_sum_surp' : sum_s, 'np_mean_surp' : mean_s, 
            'argPos' : argument, 'np_num' : number, 'definiteness' : definiteness,
        }

        

        #Plug Update
        for token in np:
            token_rows[token.i].update(np_data)
        token_rows[head.i]['is_np_head'] = True #Marks row if the token is the head noun

    return token_rows



##########
# ANALYZE
##########

def analysis(input, 
            output,
            nlp,
            accelerator,
            tokenizer,
            model):
    batch_num = 0 # Batch number
    sentence_tuples = XML_tupler(input) # Outputs (text, context) tuples w/ ID numbers, filenames, and modality

    if not sentence_tuples:
        return

    total_sentences = len(sentence_tuples)
    total_batches = math.ceil(total_sentences / CSV_BATCH)

    all_token_rows = []

    for batch_num in tqdm(range(total_batches), desc="Processing Chunk", position =1, leave = False):
        batch_start_index = batch_num * CSV_BATCH

        is_first_batch = (batch_num == 0)
        file_mode = 'w' if is_first_batch else 'a'
        write_header= is_first_batch
        
        surprisal_tuples = surprisal_calc(sentence_tuples, tokenizer, model, accelerator, batch_num) # Outputs (text, context, surprisal) tuples for each sentence

        empty_gpu_cache()
        gc.collect()

        doc_stream = spacy_streamer(surprisal_tuples, nlp) # Creates stream of SpaCy docs (one per sentence) for processing
        
        token_rows = [] # Holds CSV rows for each word/token

        for doc in tqdm(doc_stream, total=len(sentence_tuples), desc="NLP Processing", position=3, leave=False): # For each sentence
            aligned = alignment(doc, tokenizer) # Align SpaCy and LLaMa tokens
            token_surprisals = tokenize_surprisal(doc, aligned) # Assign surprisals to each word
            token_rows.extend(generate_rows(doc, token_surprisals)) # Generate and attach CSV rows for each word

        # gc.collect()
        # empty_gpu_cache()

        if token_rows: # Write file to CSV
            pd.DataFrame(token_rows).to_csv(
                output, 
                mode = file_mode,
                header = write_header,
                index = False, 
                encoding = 'utf-8-sig')

    # Cleanup #

    # del sentence_tuples, surprisal_tuples, doc_stream, token_rows 
    # gc.collect()
    # empty_gpu_cache()

        

def analyze(inputDir, outputDir, spacy_model, hf_model, overwrite = 0):
    """
    Loops through XML files in a directory (recursively) and runs analysis() for each.
    Overwrite parameter: Controls if existing files will be overwritten or skipped.
    """

    if overwrite == 1:
        print("WARNING: Overwrite is set to ON. Any existing output files with identical names to newly generated outputs will be overwritten.")

    nlp, accelerator, tokenizer, model = initialize_models(spacy_model, hf_model) # Initializes a SpaCy and HuggingFace model

    # File handling # 

    all_filepaths = get_filepaths(inputDir) # Recursively searches for XML filepaths in a directory
    if not all_filepaths:
        print("Input filepath not found")
        return
    print(f"Found {len(all_filepaths)} files.")

    # Filter for existing files
    files_to_process = [] # List of files to be processed
    files_existing = 0

    for filepath in all_filepaths:
        base_filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(base_filename)[0]
        output_filename = os.path.join(outputDir, f"{filename_no_ext}.csv")

        if overwrite == 1 or not os.path.exists(output_filename): # If overwrite == 1 or the output file does not exist (i.e., all files for overwrite, otherwise only non-existent files)
            files_to_process.append((filepath, output_filename)) # Append to files to process list

        if os.path.exists(output_filename): # If filepath exists
            files_existing += 1 # Increment counter


    if overwrite == 0:
        print(f"{files_existing} output files already exist and overwrite is OFF. Processing {len(files_to_process)} files.")
    elif overwrite == 1:
        print(f"Overwrite is ON. {files_existing} files will be overwritten and {len(files_to_process)} (including overwrites) will be processed.")

    # Make output dir

    if not os.path.exists(outputDir): # If output does not exist
        os.makedirs(outputDir) # Create output directory 
        print(f"Created output folder {outputDir}.")

    # Loop over files

    for filepath, output_filename in tqdm(files_to_process, desc="Files Processed", position = 0): # For each file


        try: 
            analysis(filepath, output_filename, nlp, accelerator, tokenizer, model) # Run analysis
        #Analysis function already clears memory
        except torch.cuda.OutOfMemoryError:
            print(f"\n CUDA OOM on {os.path.basename(filepath)}")
            with open("failed_files.txt", "a") as f:
                f.write(f"{filepath} - CUDA OOM \n")

            empty_gpu_cache()
            gc.collect()

            continue

        except Exception as e: 
            print(f"\n Error {e} on file {os.path.basename(filepath)}")
            with open("failed_files.txt", "a") as f: 
                f.write(f"{filepath} - Error: {e} \n")
                continue

    print("Done!")

#!#!#!#!
# EXECUTION 
#!#!#!#!


if __name__ == "__main__":

    start_time = time.time()

    analyze(INPUT_DIR, 
        OUTPUT_DIR,
        SPACY_MOD, 
        TRANSFORMER_MOD, 
        OVERWRITE
        )
    
    end_time = time.time()

    timer(start_time, end_time)