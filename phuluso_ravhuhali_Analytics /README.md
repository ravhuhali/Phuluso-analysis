## workflow 

* create Virtual Environment
`python -m venv venv`

* Activate the Environment

`source venv/bin/activate`

* Compile requirements.txt

`pip-compile -U requirements.in`
* Install the Dependencies

`pip install -r requirements.txt`