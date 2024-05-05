// Initial wiring: [4, 7, 1, 8, 6, 3, 5, 0, 2]
// Resulting wiring: [4, 7, 1, 8, 6, 3, 5, 0, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[1], q[4];
cx q[7], q[8];
cx q[5], q[4];
cx q[4], q[1];
cx q[1], q[0];
