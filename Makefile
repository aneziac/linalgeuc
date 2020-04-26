install:
	python3 -m venv venv
	. ./venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

clean:
	rm -rf venv
	rm -rf linalgeuc.egg-info
	find . -name '*.pyc' -delete

test:
	python tests/test_linalg.py

run-raster:
	python linalgeuc/graphics/raster.py
