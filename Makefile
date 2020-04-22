install:
	python3 -m venv venv
	. ./venv/bin/activate
	pip install -r requirements.txt
	pip install -e .

clean:
	deactivate
	rm -rf venv
	rm -rf linalgeuc.egg-info
	find . -name '*.pyc' -delete

run-raster:
	python lib/graphics/raster.py
