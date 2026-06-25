# Datasets

This directory contains processed edge-list files used by DPFairFormer.

## Files

| File | Dataset | Notes |
| --- | --- | --- |
| `Bitcoinalpha.txt` | Bitcoin-Alpha | Signed trust network. |
| `Bitcoinotc.txt` | Bitcoin-OTC | Signed trust network. |
| `WikiRfa.txt` | Wiki-RfA | Signed voting/adminship network. |
| `Slashdot.txt` | Slashdot | Signed social network. |
| `amazon_book.txt` | Amazon-Book | Processed from Amazon review data; ratings >= 4 are positive, ratings <= 2 are negative. |

## Format

Each file is read as:

```text
src dst sign_or_rating
```

For Bitcoin, Wiki-RfA, and Slashdot, positive values are positive signed edges and negative values are negative signed edges. For Amazon-Book, the loader converts ratings >= 4 to positive edges and ratings <= 2 to negative edges; rating 3 is ignored.

## Sources

- Bitcoin-Alpha: https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- Bitcoin-OTC: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- Wiki-RfA: https://snap.stanford.edu/data/wiki-RfA.html
- Slashdot: https://snap.stanford.edu/data/soc-Slashdot0811.html
- Amazon product/review data: https://snap.stanford.edu/data/amazon-meta.html

Large raw Amazon files and intermediate preprocessing artifacts are not included in this repository.
