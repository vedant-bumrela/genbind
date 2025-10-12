from src.sample import samples
from src.utils import is_valid_smiles, calc_qed, calc_sa

valid_smiles = [s for s in samples if is_valid_smiles(s)]

uniqueness = len(set(valid_smiles)) / len(valid_smiles)

qed_scores = [calc_qed(s) for s in valid_smiles]
sa_scores = [calc_sa(s) for s in valid_smiles]


valid_sa_scores = [score for score in sa_scores if score is not None]

print(f"Total Generated: {len(samples)}")
print(f"Valid SMILES: {len(valid_smiles)} ({len(valid_smiles)/len(samples)*100:.2f}%)")
print(f"Uniqueness: {uniqueness*100:.2f}%")
print(f"Average QED: {sum(qed_scores)/len(qed_scores):.2f}")

if valid_sa_scores:
    print(f"Average SA: {sum(valid_sa_scores)/len(valid_sa_scores):.2f}")
else:
    print("Average SA: N/A (no valid SA scores)")
