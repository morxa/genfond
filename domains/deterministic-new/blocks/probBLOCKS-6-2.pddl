(define (problem BLOCKS-6-2)
(:domain BLOCKS)
(:objects E F B D C A )
(:init (clear A) (ontable C) (on A D) (on D B) (on B F) (on F E) (on E C)
 (handempty))
(:goal (and (on E F) (on F A) (on A B) (on B C) (on C D)))
)