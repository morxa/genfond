; Map of the Depots:                       
; 000=111 222          
; 0*0 1*1 *22          
;------------          
; 0: depot0 area
; 1: depot1 area
; 2: depot2 area
; *: Depot access point
; =: Transit area

(define (problem storage-19)
(:domain Storage-Propositional)
(:objects
	depot0_1_1 depot0_1_2 depot0_1_3 depot0_2_1 depot0_2_2 depot0_2_3 depot1_1_1 depot1_1_2 depot1_1_3 depot1_2_1 depot1_2_2 depot1_2_3 depot2_1_1 depot2_1_2 depot2_1_3 depot2_2_1 depot2_2_2 depot2_2_3 container0_0 container0_1 container0_2 container0_3 container1_0 container1_1 container1_2 container1_3 container2_0 - storearea
	hoist0 hoist1 hoist2 - hoist
	crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 - crate
	container0 container1 container2 - container
	depot0 depot1 depot2 - depot
	loadarea transit0 - transitarea)

(:init
	(connected depot0_1_1 depot0_2_1)
	(connected depot0_1_1 depot0_1_2)
	(connected depot0_1_2 depot0_2_2)
	(connected depot0_1_2 depot0_1_3)
	(connected depot0_1_2 depot0_1_1)
	(connected depot0_1_3 depot0_2_3)
	(connected depot0_1_3 depot0_1_2)
	(connected depot0_2_1 depot0_1_1)
	(connected depot0_2_1 depot0_2_2)
	(connected depot0_2_2 depot0_1_2)
	(connected depot0_2_2 depot0_2_3)
	(connected depot0_2_2 depot0_2_1)
	(connected depot0_2_3 depot0_1_3)
	(connected depot0_2_3 depot0_2_2)
	(connected depot1_1_1 depot1_2_1)
	(connected depot1_1_1 depot1_1_2)
	(connected depot1_1_2 depot1_2_2)
	(connected depot1_1_2 depot1_1_3)
	(connected depot1_1_2 depot1_1_1)
	(connected depot1_1_3 depot1_2_3)
	(connected depot1_1_3 depot1_1_2)
	(connected depot1_2_1 depot1_1_1)
	(connected depot1_2_1 depot1_2_2)
	(connected depot1_2_2 depot1_1_2)
	(connected depot1_2_2 depot1_2_3)
	(connected depot1_2_2 depot1_2_1)
	(connected depot1_2_3 depot1_1_3)
	(connected depot1_2_3 depot1_2_2)
	(connected depot2_1_1 depot2_2_1)
	(connected depot2_1_1 depot2_1_2)
	(connected depot2_1_2 depot2_2_2)
	(connected depot2_1_2 depot2_1_3)
	(connected depot2_1_2 depot2_1_1)
	(connected depot2_1_3 depot2_2_3)
	(connected depot2_1_3 depot2_1_2)
	(connected depot2_2_1 depot2_1_1)
	(connected depot2_2_1 depot2_2_2)
	(connected depot2_2_2 depot2_1_2)
	(connected depot2_2_2 depot2_2_3)
	(connected depot2_2_2 depot2_2_1)
	(connected depot2_2_3 depot2_1_3)
	(connected depot2_2_3 depot2_2_2)
	(connected transit0 depot0_1_3)
	(connected transit0 depot1_1_1)
	(in depot0_1_1 depot0)
	(in depot0_1_2 depot0)
	(in depot0_1_3 depot0)
	(in depot0_2_1 depot0)
	(in depot0_2_2 depot0)
	(in depot0_2_3 depot0)
	(in depot1_1_1 depot1)
	(in depot1_1_2 depot1)
	(in depot1_1_3 depot1)
	(in depot1_2_1 depot1)
	(in depot1_2_2 depot1)
	(in depot1_2_3 depot1)
	(in depot2_1_1 depot2)
	(in depot2_1_2 depot2)
	(in depot2_1_3 depot2)
	(in depot2_2_1 depot2)
	(in depot2_2_2 depot2)
	(in depot2_2_3 depot2)
	(on crate0 container0_0)
	(on crate1 container0_1)
	(on crate2 container0_2)
	(on crate3 container0_3)
	(on crate4 container1_0)
	(on crate5 container1_1)
	(on crate6 container1_2)
	(on crate7 container1_3)
	(on crate8 container2_0)
	(in crate0 container0)
	(in crate1 container0)
	(in crate2 container0)
	(in crate3 container0)
	(in crate4 container1)
	(in crate5 container1)
	(in crate6 container1)
	(in crate7 container1)
	(in crate8 container2)
	(in container0_0 container0)
	(in container0_1 container0)
	(in container0_2 container0)
	(in container0_3 container0)
	(in container1_0 container1)
	(in container1_1 container1)
	(in container1_2 container1)
	(in container1_3 container1)
	(in container2_0 container2)
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
	(connected loadarea container1_1) 
	(connected container1_1 loadarea)
	(connected loadarea container1_2) 
	(connected container1_2 loadarea)
	(connected loadarea container1_3) 
	(connected container1_3 loadarea)
	(connected loadarea container2_0) 
	(connected container2_0 loadarea)  
	(connected depot0_2_2 loadarea)
	(connected loadarea depot0_2_2)
	(connected depot1_2_2 loadarea)
	(connected loadarea depot1_2_2)
	(connected depot2_2_1 loadarea)
	(connected loadarea depot2_2_1)  
	(clear depot0_1_1)
	(clear depot0_2_1)
	(clear depot0_2_3)
	(clear depot1_1_1)
	(clear depot1_1_2)
	(clear depot1_1_3)
	(clear depot1_2_1)
	(clear depot1_2_2)
	(clear depot1_2_3)
	(clear depot2_1_1)
	(clear depot2_1_2)
	(clear depot2_1_3)
	(clear depot2_2_1)
	(clear depot2_2_2)
	(clear depot2_2_3)  
	(at hoist0 depot0_1_3)
	(available hoist0)
	(at hoist1 depot0_1_2)
	(available hoist1)
	(at hoist2 depot0_2_2)
	(available hoist2))

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)
	(in crate2 depot0)
	(in crate3 depot1)
	(in crate4 depot1)
	(in crate5 depot1)
	(in crate6 depot2)
	(in crate7 depot2)
	(in crate8 depot2)))
)
