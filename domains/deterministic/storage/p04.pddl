; Map of the Depots:       
; 00   
; *0   
;---   
; 0: depot0 area
; *: Depot access point
; =: Transit area

(define (problem storage-4)
(:domain Storage-Propositional)
(:objects
	depot0_1_1 depot0_1_2 depot0_2_1 depot0_2_2 container0_0 container0_1 - storearea
	hoist0 - hoist
	crate0 crate1 - crate
	container0 - container
	depot0 - depot
	loadarea - transitarea)

(:init
	(connected depot0_1_1 depot0_2_1)
	(connected depot0_1_1 depot0_1_2)
	(connected depot0_1_2 depot0_2_2)
	(connected depot0_1_2 depot0_1_1)
	(connected depot0_2_1 depot0_1_1)
	(connected depot0_2_1 depot0_2_2)
	(connected depot0_2_2 depot0_1_2)
	(connected depot0_2_2 depot0_2_1)
	(in depot0_1_1 depot0)
	(in depot0_1_2 depot0)
	(in depot0_2_1 depot0)
	(in depot0_2_2 depot0)
	(on crate0 container0_0)
	(on crate1 container0_1)
	(in crate0 container0)
	(in crate1 container0)
	(in container0_0 container0)
	(in container0_1 container0)
	(connected loadarea container0_0) 
	(connected container0_0 loadarea)
	(connected loadarea container0_1) 
	(connected container0_1 loadarea)  
	(connected depot0_2_1 loadarea)
	(connected loadarea depot0_2_1)  
	(clear depot0_2_2)
	(clear depot0_1_2)
	(clear depot0_2_1)  
	(at hoist0 depot0_1_1)
	(available hoist0))

(:goal (and
	(in crate0 depot0)
	(in crate1 depot0)))
)
