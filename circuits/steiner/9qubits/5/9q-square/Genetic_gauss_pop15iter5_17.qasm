// Initial wiring: [0 1 2 7 4 8 5 3 6]
// Resulting wiring: [0 1 2 7 4 8 5 3 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[2];
cx q[0], q[5];
cx q[5], q[6];
cx q[8], q[3];
