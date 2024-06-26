; Map of the Depots:              
; 000=111     
; 0 * 1*      
;--------     
; 0: depot0 area
; 1: depot1 area
; *: Depot access point
; =: Transit area

(define (problem storage-13)
(:domain Storage-Propositional)
(:objects
	depot0_1_1 depot0_1_2 depot0_1_3 depot0_2_1 depot0_2_3 depot1_1_1 depot1_1_2 depot1_1_3 depot1_2_1 depot1_2_2 container0_0 container0_1 container0_2 container0_3 container1_0 - storearea
	hoist0 - hoist
	crate0 crate1 crate2 crate3 crate4 - crate
	container0 container1 - container
	depot0 depot1 - depot
	loadarea transit0 - transitarea)

(:init
	(connected depot0_1_1 depot0_2_1)
	(connected depot0_1_1 depot0_1_2)
	(connected depot0_1_2 depot0_1_3)
	(connected depot0_1_2 depot0_1_1)
	(connected depot0_1_3 depot0_2_3)
	(connected depot0_1_3 depot0_1_2)
	(connected depot0_2_1 depot0_1_1)
	(connected depot0_2_3 depot0_1_3)
	(connected depot1_1_1 depot1_2_1)
	(connected depot1_1_1 depot1_1_2)
	(connected depot1_1_2 depot1_2_2)
	(connected depot1_1_2 depot1_1_3)
	(connected depot1_1_2 depot1_1_1)
	(connected depot1_1_3 depot1_1_2)
	(connected depot1_2_1 depot1_1_1)
	(connected depot1_2_1 depot1_2_2)
	(connected depot1_2_2 depot1_1_2)
	(connected depot1_2_2 depot1_2_1)
	(connected transit0 depot0_1_3)
	(connected transit0 depot1_1_1)
	(in depot0_1_1 depot0)
	(in depot0_1_2 depot0)
	(in depot0_1_3 depot0)
	(in depot0_2_1 depot0)
	(in depot0_2_3 depot0)
	(in depot1_1_1 depot1)
	(in depot1_1_2 depot1)
	(in depot1_1_3 depot1)
	(in depot1_2_1 depot1)
	(in depot1_2_2 depot1)
	(on crate0 container0_0)
	(on crate1 container0_1)
	(on crate2 container0_2)
	(on crate3 container0_3)
	(on crate4 container1_0)
	(in crate0 container0)
	(in crate1 container0)
	(in crate2 container0)
	(in crate3 container0)
	(in crate4 container1)
	(in container0_0 container0)
	(in container0_1 container0)
	(in container0_2 container0)
	(in container0_3 container0)
	(in container1_0 container1)
	(connected loadarea container0_0) 
	(connected container0_0 loadarea)
	(connected loadarea container0_1) 
	(connected container0_1 loadarea)
	(connected loadarea container0_2) 
	(connected container0_2 loadarea)
	(connected loadarea container0_3) 
	(connected container0_3 loadarea)
	(connected loadarea container1_0) 
	(connected container1_0 loadarea)  
	(connected depot0_2_3 loadarea)
	(connected loadarea depot0_2_3)
	(connected depot1_2_2 loadarea)
	(connected loadarea depot1_2_2)  
	(clear depot0_1_1)
	(clear depot0_1_2)
	(clear depot0_2_3)
	(clear depot0_2_1)
	(clear depot1_1_1)
	(clear depot1_1_2)
	(clear depot1_1_3)
	(clear depot1_2_1)
	(clear depot1_2_2)  
	(at hoist0 depot0_1_3)
	(available hoist0))

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)
	(in crate2 depot1)
	(in crate3 depot1)
	(in crate4 depot1)))
)
