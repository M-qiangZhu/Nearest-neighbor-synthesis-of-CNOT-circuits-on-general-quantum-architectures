// Initial wiring: [4, 7, 3, 0, 2, 1, 8, 6, 5]
// Resulting wiring: [4, 7, 3, 0, 2, 1, 8, 6, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[2], q[1];
cx q[5], q[4];
cx q[4], q[3];
cx q[3], q[2];
