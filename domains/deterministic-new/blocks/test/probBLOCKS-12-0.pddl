(define (problem BLOCKS-12-0)
(:domain BLOCKS)
(:objects I D B E K G A F C J L H )
(:init (clear H) (clear L) (clear J) (ontable C) (ontable F) (ontable J)
 (on H A) (on A G) (on G K) (on K E) (on E B) (on B D) (on D I) (on I C)
 (on L F) (handempty))
(:goal (and (on I C) (on C B) (on B L) (on L D) (on D J) (on J E) (on E K)
            (on K F) (on F A) (on A H) (on H G)))
)