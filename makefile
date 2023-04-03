compile:
	@rm -f requirements*.txt
	@pip-compile requirements.in
	@pip-compile requirements-dev.in

install:
	@pip install \
		-r requirements.txt \
		-r requirements-dev.txt
sync:
	@pip-sync requirements*.txt