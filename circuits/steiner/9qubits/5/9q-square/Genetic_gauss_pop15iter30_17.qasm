// Initial wiring: [1 5 2 4 3 0 6 7 8]
// Resulting wiring: [1 5 2 4 3 0 6 7 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[6], q[7];
cx q[2], q[1];
cx q[1], q[4];
cx q[5], q[6];
