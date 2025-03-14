; Map of the Depots:                                             
; 0000=1111 2222 3333 444                    
; *00  *111 22*  *33  444                    
;                     4*4                    
;------------------------                    
; 0: depot0 area
; 1: depot1 area
; 2: depot2 area
; 3: depot3 area
; 4: depot4 area
; *: Depot access point
; =: Transit area

(define (problem storage-29)
(:domain Storage-Propositional)
(:objects
	depot0_1_1 depot0_1_2 depot0_1_3 depot0_1_4 depot0_2_1 depot0_2_2 depot0_2_3 depot1_1_1 depot1_1_2 depot1_1_3 depot1_1_4 depot1_2_1 depot1_2_2 depot1_2_3 depot1_2_4 depot2_1_1 depot2_1_2 depot2_1_3 depot2_1_4 depot2_2_1 depot2_2_2 depot2_2_3 depot3_1_1 depot3_1_2 depot3_1_3 depot3_1_4 depot3_2_1 depot3_2_2 depot3_2_3 depot4_1_1 depot4_1_2 depot4_1_3 depot4_2_1 depot4_2_2 depot4_2_3 depot4_3_1 depot4_3_2 depot4_3_3 container0_0 container0_1 container0_2 container0_3 container1_0 container1_1 container1_2 container1_3 container2_0 container2_1 container2_2 container2_3 container3_0 container3_1 container3_2 container3_3 container4_0 container4_1 container4_2 - storearea
	hoist0 hoist1 hoist2 hoist3 hoist4 - hoist
	crate0 crate1 crate2 crate3 crate4 crate5 crate6 crate7 crate8 crate9 crate10 crate11 crate12 crate13 crate14 crate15 crate16 crate17 crate18 - crate
	container0 container1 container2 container3 container4 - container
	depot0 depot1 depot2 depot3 depot4 - depot
	loadarea transit0 - transitarea)

(:init
	(connected depot0_1_1 depot0_2_1)
	(connected depot0_1_1 depot0_1_2)
	(connected depot0_1_2 depot0_2_2)
	(connected depot0_1_2 depot0_1_3)
	(connected depot0_1_2 depot0_1_1)
	(connected depot0_1_3 depot0_2_3)
	(connected depot0_1_3 depot0_1_4)
	(connected depot0_1_3 depot0_1_2)
	(connected depot0_1_4 depot0_1_3)
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
	(connected depot1_1_3 depot1_1_4)
	(connected depot1_1_3 depot1_1_2)
	(connected depot1_1_4 depot1_2_4)
	(connected depot1_1_4 depot1_1_3)
	(connected depot1_2_1 depot1_1_1)
	(connected depot1_2_1 depot1_2_2)
	(connected depot1_2_2 depot1_1_2)
	(connected depot1_2_2 depot1_2_3)
	(connected depot1_2_2 depot1_2_1)
	(connected depot1_2_3 depot1_1_3)
	(connected depot1_2_3 depot1_2_4)
	(connected depot1_2_3 depot1_2_2)
	(connected depot1_2_4 depot1_1_4)
	(connected depot1_2_4 depot1_2_3)
	(connected depot2_1_1 depot2_2_1)
	(connected depot2_1_1 depot2_1_2)
	(connected depot2_1_2 depot2_2_2)
	(connected depot2_1_2 depot2_1_3)
	(connected depot2_1_2 depot2_1_1)
	(connected depot2_1_3 depot2_2_3)
	(connected depot2_1_3 depot2_1_4)
	(connected depot2_1_3 depot2_1_2)
	(connected depot2_1_4 depot2_1_3)
	(connected depot2_2_1 depot2_1_1)
	(connected depot2_2_1 depot2_2_2)
	(connected depot2_2_2 depot2_1_2)
	(connected depot2_2_2 depot2_2_3)
	(connected depot2_2_2 depot2_2_1)
	(connected depot2_2_3 depot2_1_3)
	(connected depot2_2_3 depot2_2_2)
	(connected depot3_1_1 depot3_2_1)
	(connected depot3_1_1 depot3_1_2)
	(connected depot3_1_2 depot3_2_2)
	(connected depot3_1_2 depot3_1_3)
	(connected depot3_1_2 depot3_1_1)
	(connected depot3_1_3 depot3_2_3)
	(connected depot3_1_3 depot3_1_4)
	(connected depot3_1_3 depot3_1_2)
	(connected depot3_1_4 depot3_1_3)
	(connected depot3_2_1 depot3_1_1)
	(connected depot3_2_1 depot3_2_2)
	(connected depot3_2_2 depot3_1_2)
	(connected depot3_2_2 depot3_2_3)
	(connected depot3_2_2 depot3_2_1)
	(connected depot3_2_3 depot3_1_3)
	(connected depot3_2_3 depot3_2_2)
	(connected depot4_1_1 depot4_2_1)
	(connected depot4_1_1 depot4_1_2)
	(connected depot4_1_2 depot4_2_2)
	(connected depot4_1_2 depot4_1_3)
	(connected depot4_1_2 depot4_1_1)
	(connected depot4_1_3 depot4_2_3)
	(connected depot4_1_3 depot4_1_2)
	(connected depot4_2_1 depot4_1_1)
	(connected depot4_2_1 depot4_3_1)
	(connected depot4_2_1 depot4_2_2)
	(connected depot4_2_2 depot4_1_2)
	(connected depot4_2_2 depot4_3_2)
	(connected depot4_2_2 depot4_2_3)
	(connected depot4_2_2 depot4_2_1)
	(connected depot4_2_3 depot4_1_3)
	(connected depot4_2_3 depot4_3_3)
	(connected depot4_2_3 depot4_2_2)
	(connected depot4_3_1 depot4_2_1)
	(connected depot4_3_1 depot4_3_2)
	(connected depot4_3_2 depot4_2_2)
	(connected depot4_3_2 depot4_3_3)
	(connected depot4_3_2 depot4_3_1)
	(connected depot4_3_3 depot4_2_3)
	(connected depot4_3_3 depot4_3_2)
	(connected transit0 depot0_1_4)
	(connected transit0 depot1_1_1)
	(in depot0_1_1 depot0)
	(in depot0_1_2 depot0)
	(in depot0_1_3 depot0)
	(in depot0_1_4 depot0)
	(in depot0_2_1 depot0)
	(in depot0_2_2 depot0)
	(in depot0_2_3 depot0)
	(in depot1_1_1 depot1)
	(in depot1_1_2 depot1)
	(in depot1_1_3 depot1)
	(in depot1_1_4 depot1)
	(in depot1_2_1 depot1)
	(in depot1_2_2 depot1)
	(in depot1_2_3 depot1)
	(in depot1_2_4 depot1)
	(in depot2_1_1 depot2)
	(in depot2_1_2 depot2)
	(in depot2_1_3 depot2)
	(in depot2_1_4 depot2)
	(in depot2_2_1 depot2)
	(in depot2_2_2 depot2)
	(in depot2_2_3 depot2)
	(in depot3_1_1 depot3)
	(in depot3_1_2 depot3)
	(in depot3_1_3 depot3)
	(in depot3_1_4 depot3)
	(in depot3_2_1 depot3)
	(in depot3_2_2 depot3)
	(in depot3_2_3 depot3)
	(in depot4_1_1 depot4)
	(in depot4_1_2 depot4)
	(in depot4_1_3 depot4)
	(in depot4_2_1 depot4)
	(in depot4_2_2 depot4)
	(in depot4_2_3 depot4)
	(in depot4_3_1 depot4)
	(in depot4_3_2 depot4)
	(in depot4_3_3 depot4)
	(on crate0 container0_0)
	(on crate1 container0_1)
	(on crate2 container0_2)
	(on crate3 container0_3)
	(on crate4 container1_0)
	(on crate5 container1_1)
	(on crate6 container1_2)
	(on crate7 container1_3)
	(on crate8 container2_0)
	(on crate9 container2_1)
	(on crate10 container2_2)
	(on crate11 container2_3)
	(on crate12 container3_0)
	(on crate13 container3_1)
	(on crate14 container3_2)
	(on crate15 container3_3)
	(on crate16 container4_0)
	(on crate17 container4_1)
	(on crate18 container4_2)
	(in crate0 container0)
	(in crate1 container0)
	(in crate2 container0)
	(in crate3 container0)
	(in crate4 container1)
	(in crate5 container1)
	(in crate6 container1)
	(in crate7 container1)
	(in crate8 container2)
	(in crate9 container2)
	(in crate10 container2)
	(in crate11 container2)
	(in crate12 container3)
	(in crate13 container3)
	(in crate14 container3)
	(in crate15 container3)
	(in crate16 container4)
	(in crate17 container4)
	(in crate18 container4)
	(in container0_0 container0)
	(in container0_1 container0)
	(in container0_2 container0)
	(in container0_3 container0)
	(in container1_0 container1)
	(in container1_1 container1)
	(in container1_2 container1)
	(in container1_3 container1)
	(in container2_0 container2)
	(in container2_1 container2)
	(in container2_2 container2)
	(in container2_3 container2)
	(in container3_0 container3)
	(in container3_1 container3)
	(in container3_2 container3)
	(in container3_3 container3)
	(in container4_0 container4)
	(in container4_1 container4)
	(in container4_2 container4)
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
	(connected loadarea container2_1) 
	(connected container2_1 loadarea)
	(connected loadarea container2_2) 
	(connected container2_2 loadarea)
	(connected loadarea container2_3) 
	(connected container2_3 loadarea)
	(connected loadarea container3_0) 
	(connected container3_0 loadarea)
	(connected loadarea container3_1) 
	(connected container3_1 loadarea)
	(connected loadarea container3_2) 
	(connected container3_2 loadarea)
	(connected loadarea container3_3) 
	(connected container3_3 loadarea)
	(connected loadarea container4_0) 
	(connected container4_0 loadarea)
	(connected loadarea container4_1) 
	(connected container4_1 loadarea)
	(connected loadarea container4_2) 
	(connected container4_2 loadarea)  
	(connected depot0_2_1 loadarea)
	(connected loadarea depot0_2_1)
	(connected depot1_2_1 loadarea)
	(connected loadarea depot1_2_1)
	(connected depot2_2_3 loadarea)
	(connected loadarea depot2_2_3)
	(connected depot3_2_1 loadarea)
	(connected loadarea depot3_2_1)
	(connected depot4_3_2 loadarea)
	(connected loadarea depot4_3_2)  
	(clear depot0_2_3)
	(clear depot0_1_2)
	(clear depot0_1_3)
	(clear depot0_1_4)
	(clear depot0_2_1)
	(clear depot0_2_2)
	(clear depot1_1_1)
	(clear depot1_1_2)
	(clear depot1_1_3)
	(clear depot1_1_4)
	(clear depot1_2_1)
	(clear depot1_2_2)
	(clear depot1_2_3)
	(clear depot1_2_4)
	(clear depot2_1_1)
	(clear depot2_1_2)
	(clear depot2_1_3)
	(clear depot2_1_4)
	(clear depot2_2_1)
	(clear depot2_2_2)
	(clear depot2_2_3)
	(clear depot3_1_1)
	(clear depot3_1_2)
	(clear depot3_2_2)
	(clear depot3_1_4)
	(clear depot3_2_1)
	(clear depot4_3_3)
	(clear depot4_1_2)
	(clear depot4_3_2)
	(clear depot4_2_1)
	(clear depot4_2_2)
	(clear depot4_2_3)
	(clear depot4_3_1)  
	(at hoist0 depot0_1_1)
	(available hoist0)
	(at hoist1 depot3_1_3)
	(available hoist1)
	(at hoist2 depot3_2_3)
	(available hoist2)
	(at hoist3 depot4_1_1)
	(available hoist3)
	(at hoist4 depot4_1_3)
	(available hoist4))

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)
	(in crate2 depot0)
	(in crate3 depot0)
	(in crate4 depot1)
	(in crate5 depot1)
	(in crate6 depot1)
	(in crate7 depot1)
	(in crate8 depot2)
	(in crate9 depot2)
	(in crate10 depot2)
	(in crate11 depot2)
	(in crate12 depot3)
	(in crate13 depot3)
	(in crate14 depot3)
	(in crate15 depot3)
	(in crate16 depot4)
	(in crate17 depot4)
	(in crate18 depot4)))
)
