// EXPECTED_REWIRING [0 7 2 3 4 5 6 1 8 9 10 11 12 13 14 15 16 17 18 19]
// CURRENT_REWIRING [0 8 2 3 4 5 6 1 11 9 10 14 17 12 13 15 16 7 18 19]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(0.10344064106915161*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4189783790674746*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-3.075736053375836*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[18];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(1.674236967864048*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4189783790674746*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.7843290499389812*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(2.0779896335268964*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
rz(-2.381184772407101*pi) q[17];
cz q[17], q[12];
rz(1.6366529270088535*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[17];
cz q[17], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[12];
rz(0.59368010174542*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.958108965734335*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4741817804854853*pi) q[8];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[8];
rz(3.141592653589793*pi) q[18];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-3.075736053375836*pi) q[7];
cz q[8], q[2];
rz(-1.164582056715158*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.16538560610687725*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[6];
rz(-2.087802470758894*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.3844841619731474*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.2762476260936904*pi) q[14];
rz(-1.784329049938982*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268955*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.381184772407101*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.3582145202377345*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.044053322430445*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.0678547836725985*pi) q[12];
rz(1.837488127823869*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.5543755142764981*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(-1.5707963267948966*pi) q[12];
rz(2.9374451697377317*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(-1.3709242198536704*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0064329914606325*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[14], q[13];
rz(1.3011167383896503*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(1.4005291277933123*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.14123624548958238*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.7103440603859537*pi) q[12];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[7];
cz q[12], q[7];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
cz q[2], q[7];
rz(-1.1645820567151592*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.16538560610687794*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.730367851897572*pi) q[11];
cz q[11], q[18];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rz(-2.4952420389150767*pi) q[7];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(1.467355685725745*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(2.217146941469614*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.1645820567151592*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687794*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.6269272108284194*pi) q[14];
rz(1.4564375502462918*pi) q[17];
rx(1.5707963267948966*pi) q[17];
rz(1.426995486693993*pi) q[17];
rx(-1.5707963267948966*pi) q[17];
rz(-3.059616298134544*pi) q[17];
rz(3.141592653589793*pi) q[18];
