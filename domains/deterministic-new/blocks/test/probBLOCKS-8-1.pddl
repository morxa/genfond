(define (problem BLOCKS-8-1)
(:domain BLOCKS)
(:objects B A G C F D H E )
(:init (clear E) (clear H) (clear D) (clear F) (ontable C) (ontable G)
 (ontable D) (ontable F) (on E C) (on H A) (on A B) (on B G) (handempty))
(:goal (and (on C D) (on D B) (on B G) (on G F) (on F H) (on H A) (on A E)))
)