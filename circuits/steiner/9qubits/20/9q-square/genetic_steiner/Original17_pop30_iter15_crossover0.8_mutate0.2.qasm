// Initial wiring: [6, 1, 0, 2, 5, 3, 4, 7, 8]
// Resulting wiring: [6, 1, 0, 2, 5, 3, 4, 7, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[0];
cx q[2], q[1];
cx q[5], q[4];
cx q[4], q[1];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[1];
cx q[5], q[4];
cx q[8], q[3];
cx q[3], q[2];
cx q[8], q[3];
cx q[5], q[6];
cx q[0], q[5];
cx q[0], q[1];
cx q[5], q[6];
cx q[1], q[2];
cx q[5], q[0];
