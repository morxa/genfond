; Map of the Depots:    
; * 
;-- 
; 0: depot0 area
; *: Depot access point
; =: Transit area

(define (problem storage-1)
(:domain Storage-Propositional)
(:objects
	depot0_1_1 container0_0 - storearea
	hoist0 - hoist
	crate0 - crate
	container0 - container
	depot0 - depot
	loadarea - transitarea)

(:init
	(in depot0_1_1 depot0)
	(on crate0 container0_0)
	(in crate0 container0)
	(in container0_0 container0)
	(connected loadarea container0_0) 
	(connected container0_0 loadarea)  
	(connected depot0_1_1 loadarea)
	(connected loadarea depot0_1_1)    
	(at hoist0 depot0_1_1)
	(available hoist0))

(:goal (and
	(in crate0 depot0)))
)
