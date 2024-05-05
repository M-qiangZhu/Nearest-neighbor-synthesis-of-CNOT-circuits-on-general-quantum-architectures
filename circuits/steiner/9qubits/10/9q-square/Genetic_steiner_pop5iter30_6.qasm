// Initial wiring: [4, 6, 8, 3, 2, 5, 1, 0, 7]
// Resulting wiring: [4, 6, 8, 3, 2, 5, 1, 0, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[1], q[2];
cx q[3], q[4];
cx q[2], q[3];
cx q[0], q[5];
cx q[4], q[7];
cx q[3], q[4];
cx q[2], q[3];
cx q[4], q[7];
cx q[8], q[7];
cx q[8], q[3];
cx q[3], q[2];
cx q[8], q[3];
cx q[2], q[1];
cx q[3], q[2];
cx q[2], q[3];
