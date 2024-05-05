
from ezQpy import *

account = Account(login_key='4eda79c7093fd23cd76c5a4d3aca5615', machine_name='ClosedBetaQC')

qcis_circuit = '''
H Q7
X Q1
H Q1
CZ Q7 Q1
H Q1
M Q7
M Q1
'''

query_id = account.submit_job(qcis_circuit)

print(query_id)

if query_id:
    result = account.query_experiment(query_id, max_wait_time=360000)

    print(result)

    value = result
    # print(value)

    f = open("results.txt", 'w')
    f.write(str(value))
    f.close()
else:
    print("error")

# res = account.download_config()
# print(res)