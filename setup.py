from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
	name="house-price-lab",
	version="0.1.0",
	description="An interactive house-price prediction and model comparison lab",
	long_description=(ROOT / "README.md").read_text(encoding="utf-8")
	if (ROOT / "README.md").exists()
	else "",
	package_dir={"": "src"},
	packages=find_packages(where="src"),
	include_package_data=True,
	python_requires=">=3.9",
	install_requires=(ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
)
