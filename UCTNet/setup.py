from setuptools import setup, find_namespace_packages

setup(name='uctnet',
      packages=find_namespace_packages(include=["uctnet", "uctnet.*"]),
      version='0.0.1',
      install_requires=[
           "wandb",
           "thop",
           "timm",
           "einops"
      ],
      entry_points={
          'console_scripts': [
              'UCTNet_train = uctnet.run.run_training:main',
              'UCTNet_BCV = uctnet.dataset_conversion.Task017_BeyondCranialVaultAbdominalOrganSegmentation:main',
              'UCTNet_ACDC = uctnet.dataset_conversion.Task027_AutomaticCardiacDetectionChallenge:main',
              'UCTNet_ISIC = uctnet.dataset_conversion.Task100_ISIC2018:main',
              'UCTNet_predict = uctnet.inference.predict_simple:main',
              
          ],
      },
      )
