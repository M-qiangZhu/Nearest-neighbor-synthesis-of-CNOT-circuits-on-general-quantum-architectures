// EXPECTED_REWIRING [0 4 2 8 7 1 6 5 3]
// CURRENT_REWIRING [4 5 2 7 6 1 3 0 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
cz q[3], q[8];
rz(2.5479125518443735*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.18348368785545818*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.6674108731043074*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(0.59368010174542*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.958108965734335*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.4741817804854853*pi) q[6];
rz(1.3572636036508112*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.8103884456122045*pi) q[7];
cz q[7], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.9280599304457073*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.0636030200628968*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.760407881182692*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
cz q[2], q[1];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.495242038915076*pi) q[3];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(1.9770105968746368*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.976207047482916*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.1645820567151572*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068784*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.1638553900637731*pi) q[8];
cz q[7], q[8];
rz(3.141592653589793*pi) q[1];
cz q[5], q[4];
rx(1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(-3.0381520125206416*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.298994492700357*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.0636030200628965*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(1.6366529270088535*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.7604078811826915*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[4];
rz(2.487346972302436*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.217366411692774*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[2];
rz(-0.5936801017454264*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.18348368785545865*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.0036142183255263*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0636030200628972*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-0.9033854536905833*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.7604078811826904*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(0.24271325173163064*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.261599837637768*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.244269372863136*pi) q[3];
cz q[2], q[3];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.977010596874633*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.976207047482916*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[5];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.08546176955627*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[3], q[4];
rz(-1.6785208328251482*pi) q[8];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[8];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.1645820567151595*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.1653856061068779*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.5146654427613733*pi) q[7];
rz(2.2660200306944276*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4054950215890891*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.058955949066339*pi) q[0];
rz(2.8387332156875913*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.34034411228772216*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4778724828009526*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[5];
cz q[0], q[5];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-0.17007685327686595*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9737760110429252*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
rz(1.0728568488664403*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.706089576074524*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.3508349036604637*pi) q[5];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(2.9085453426417063*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.4275967720542733*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.30431607700153585*pi) q[4];
cz q[5], q[4];
rz(-2.948917278310855*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(2.876382405562515*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.7519741561308915*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(2.5694846711518142*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[1];
rx(3.141592653589793*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rz(-0.9244457121201783*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(-1.1645820567151588*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687789*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.626927210828419*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6463506146747173*pi) q[6];
rx(1.5707963267948966*pi) q[7];
rz(2.9715158003129254*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
