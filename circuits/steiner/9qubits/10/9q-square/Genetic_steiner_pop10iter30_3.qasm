// Initial wiring: [0, 8, 2, 5, 7, 1, 6, 3, 4]
// Resulting wiring: [0, 8, 2, 5, 7, 1, 6, 3, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[3], q[4];
cx q[4], q[5];
cx q[3], q[4];
cx q[0], q[5];
cx q[5], q[6];
cx q[4], q[5];
cx q[4], q[7];
cx q[6], q[7];
cx q[1], q[4];
cx q[5], q[6];
cx q[4], q[7];
cx q[7], q[8];
cx q[4], q[7];
cx q[7], q[8];
cx q[1], q[0];
