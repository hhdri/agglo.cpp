ensure-glove:
	# https://nlp.stanford.edu/data/glove.6B.zip
	sha256sum -c sha256checksums.txt

gen-e2e-expected: ensure-glove
	python scripts/gen_e2e_expected.py glove.6B.50d.txt 250 50 > expected.txt