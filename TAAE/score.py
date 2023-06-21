import sys
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.scores import accuracy
from rouge_score import rouge_scorer


def divide_inference(inference_file):
    with open(inference_file, 'r', encoding='utf-8') as cand_file:
        inferences = [line.strip() for line in cand_file.readlines()]

        ref_file = open('checkpoints/output/test_reference.txt', 'w')
        cand_file = open('checkpoints/output/test_candidate.txt', 'w')

        for i in range(int(len(inferences)/3)):
            ref_file.write(inferences[i*3] + '\n')
            cand_file.write(inferences[i*3 + 1] + '\n')

        ref_file.close()
        cand_file.close()

    return ref_file.name, cand_file.name

def calculate_bleu(reference_file, candidate_file):
    with open(reference_file, 'r', encoding='utf-8') as ref_file, open(candidate_file, 'r', encoding='utf-8') as cand_file:
        references = [line.strip() for line in ref_file.readlines()]
        candidates = [line.strip() for line in cand_file.readlines()]

    assert len(references) == len(candidates), "Reference and candidate files must have the same number of lines."

    total_score = 0
    num_sentences = len(references)
    smoothing = SmoothingFunction()

    for ref, cand in zip(references, candidates):
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
        total_score += score

    average_score = total_score / num_sentences
    return average_score

def calculate_rouge_l(reference_file, candidate_file):
    with open(reference_file, 'r', encoding='utf-8') as ref_file, open(candidate_file, 'r', encoding='utf-8') as cand_file:
        references = [line.strip() for line in ref_file.readlines()]
        candidates = [line.strip() for line in cand_file.readlines()]

    assert len(references) == len(
        candidates), "Reference and candidate files must have the same number of lines."

    total_score = 0
    num_sentences = len(references)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for ref, cand in zip(references, candidates):

        score = scorer.score(ref, cand)['rougeL'].fmeasure
        total_score += score

    average_score = total_score / num_sentences
    return average_score


def main(args):
    reference, candidate = divide_inference(args.inference)
    bleu_score = calculate_bleu(reference, candidate)
    rouge_l_score = calculate_rouge_l(reference, candidate)
    print(f"Average BLEU score: {bleu_score:.4f}")
    print(f"Average ROUGE-L score: {rouge_l_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU score and Rouge score between two text files.")
    parser.add_argument("--inference", type=str, required=True, help="Path to the reference text file.")
    args = parser.parse_args()
    main(args)