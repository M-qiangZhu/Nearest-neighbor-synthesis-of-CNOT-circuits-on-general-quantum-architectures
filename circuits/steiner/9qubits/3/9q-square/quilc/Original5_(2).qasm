// EXPECTED_REWIRING [0 1 3 2 6 4 5 7 8]
// CURRENT_REWIRING [0 1 3 2 6 4 5 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[1];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[5];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(3.141592653589793*pi) q[5];
