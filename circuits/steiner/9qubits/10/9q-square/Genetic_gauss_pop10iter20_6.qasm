// Initial wiring: [0 7 2 3 4 5 6 8 1]
// Resulting wiring: [0 6 2 4 3 5 7 8 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[7], q[6];
cx q[0], q[1];
cx q[4], q[5];
cx q[4], q[5];
cx q[7], q[4];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[0], q[5];
cx q[7], q[8];
