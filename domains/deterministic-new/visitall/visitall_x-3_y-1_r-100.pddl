(define (problem visitall_x-3_y-1_r-100)
(:domain grid-visit-all)
(:objects 
	loc-x0-y0
	loc-x1-y0
	loc-x2-y0
- place 
        
)
(:init
	(at-robot loc-x0-y0)
	(visited loc-x0-y0)
	(connected loc-x0-y0 loc-x1-y0)
 	(connected loc-x1-y0 loc-x0-y0)
 	(connected loc-x1-y0 loc-x2-y0)
 	(connected loc-x2-y0 loc-x1-y0)
 
)
(:goal
(and 
	(visited loc-x0-y0)
	(visited loc-x1-y0)
	(visited loc-x2-y0)
)
)
)
