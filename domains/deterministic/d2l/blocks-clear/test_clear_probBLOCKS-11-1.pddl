(define (problem blocks-11-1)
(:domain blocks)
(:objects b c e a h k i g d f j )
(:init (clear j) (clear f) (clear d) (clear g) (ontable i) (ontable k)
 (ontable h) (ontable a) (on j i) (on f e) (on e k) (on d c) (on c h) (on g b)
 (on b a) (handempty))
(:goal (and (clear a)))

)