// EXPECTED_REWIRING [0 1 2 5 12 11 6 7 9 8 3 4 10 13 14 15]
// CURRENT_REWIRING [0 1 2 7 12 11 6 4 9 8 3 5 10 13 14 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(0.1382865274660009*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.326068754692569*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.330924369330208*pi) q[6];
rz(-2.117050410856305*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9696110278874099*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[6];
rz(-1.0313346180673755*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-2.164476428540317*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.1834836878554581*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.5400383806994418*pi) q[5];
rz(-1.3741304789600817*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.2021111577751784*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.6365639681087343*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.376391192381716*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[9], q[6];
rz(2.3312567220251355*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.3409874737914445*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.2132066213797605*pi) q[7];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.592452122213544*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.2050299505887283*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.597301185393017*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9239610091427011*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(1.467355685725755*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-2.82257681310538*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.0512084314439623*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[4];
rz(2.1839370606072843*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[3], q[4];
rx(-1.5707963267948966*pi) q[5];
rz(-0.021655795418646484*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(3.141592653589793*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(1.0512084314439623*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
rz(-0.31901584048441256*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
