(define (problem BLOCKS-14-0)
(:domain BLOCKS)
(:objects I D B L C K M H J N E F G A )
(:init (clear A) (clear G) (clear F) (ontable E) (ontable N) (ontable F)
 (on A J) (on J H) (on H M) (on M K) (on K C) (on C L) (on L B) (on B E)
 (on G D) (on D I) (on I N) (handempty))
(:goal (and (on E L) (on L F) (on F B) (on B J) (on J I) (on I N) (on N C)
            (on C K) (on K G) (on G D) (on D M) (on M A) (on A H)))
)