# -*- coding: utf-8 -*-

"""
    @Author kungfu
    @Date 2023/10/30 23:54
    @Describe 
    @Version 1.0
"""

from qiskit import IBMQ

# Configure an IBM Quantum Experience account
IBMQ.save_account("86d1f23f0a57fefbd11e508ebf722ef6237238f854de5326079f82119b78904647bc56227ee39f6bf4433ba4f251d34e2a0afc12e60436f22ab3d1b8f9ebbafe")

# Load the IBMQ backend
IBMQ.load_account()

# Get the "quito" device
provider = IBMQ.get_provider(hub='ibm-q')
quito = provider.get_backend("ibmq_quito")

# Get device parameter information
properties = quito.properties()

# Print all parameter information
print(properties)
