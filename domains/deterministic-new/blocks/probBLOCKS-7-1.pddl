(define (problem BLOCKS-7-1)
(:domain BLOCKS)
(:objects E B D F G C A )
(:init (clear A) (clear C) (ontable G) (ontable F) (on A G) (on C D) (on D B)
 (on B E) (on E F) (handempty))
(:goal (and (on A E) (on E B) (on B F) (on F G) (on G C) (on C D)))
)