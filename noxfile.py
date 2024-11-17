import nox

locations = "src", "main", "noxfile.py"

PYTHON_VERSIONS = ["3.9.16"]


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(tags=["style", "fix"])
def black(session):
    session.install("black")
    session.run("black", ".")


@nox.session(python=PYTHON_VERSIONS, tags=["style", "fix"])
def isort(session):
    session.install("isort")
    session.run("isort", ".")


@nox.session(python=PYTHON_VERSIONS, tags=["style", "fix"])
def pylint(session):
    session.install("pylint")
    session.run("pylint", *locations)


@nox.session(python=PYTHON_VERSIONS)
def run(session):
    session.run("python", "-m", "main.main", external=False)
