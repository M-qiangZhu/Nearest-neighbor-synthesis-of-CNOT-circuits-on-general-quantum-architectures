// Initial wiring: [0, 7, 5, 2, 6, 4, 1, 8, 3]
// Resulting wiring: [0, 7, 5, 2, 6, 4, 1, 8, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[7], q[6];
cx q[7], q[4];
cx q[2], q[5];
cx q[0], q[3];
cx q[2], q[8];
