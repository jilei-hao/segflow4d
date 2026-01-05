"""Packaging configuration for segflow4d."""

from pathlib import Path
from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else "segflow4d"


setup(
	name="segflow4d",
	version="0.1.0",
	description="Create 4D segmentation from sparse segmentations using image registration",
	long_description=README,
	long_description_content_type="text/markdown",
	author="",
	python_requires=">=3.9",
	package_dir={"": "src"},
	packages=find_packages(where="src"),
	py_modules=["run_fire_ants"],
	install_requires=[
		"torch>=1.8",
		"SimpleITK",
		"fireants",
		"matplotlib",
        "vtk",
	],
	entry_points={
		"console_scripts": [
            "segflow4d=main:main",
			"run-fireants=registration.fireants.run_fireants:main",
            "create-reference-mask=processing.cli.create_reference_mask:main",
            "create-tp-images=processing.cli.create_tp_images:main",
            "resample-to-reference=processing.cli.resample_to_reference:main",
		],
	},
)
