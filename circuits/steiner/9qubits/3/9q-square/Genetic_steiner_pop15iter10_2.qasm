// Initial wiring: [4, 7, 5, 6, 1, 2, 3, 0, 8]
// Resulting wiring: [4, 7, 5, 6, 1, 2, 3, 0, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[3], q[2];
cx q[4], q[3];
