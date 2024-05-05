// Initial wiring: [5, 0, 3, 1, 2, 6, 8, 4, 7]
// Resulting wiring: [5, 0, 3, 1, 2, 6, 8, 4, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[1], q[0];
cx q[5], q[4];
cx q[6], q[5];
cx q[7], q[6];
cx q[7], q[4];
cx q[6], q[5];
cx q[4], q[1];
cx q[8], q[7];
cx q[7], q[8];
cx q[4], q[7];
cx q[7], q[8];
cx q[2], q[3];
cx q[1], q[4];
cx q[4], q[7];
cx q[0], q[1];
