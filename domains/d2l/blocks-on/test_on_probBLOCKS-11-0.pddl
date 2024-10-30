(define (problem p-11-0)
(:domain blocks)
(:objects f a k h g e d i c j b )
(:init (clear b) (clear j) (clear c) (ontable i) (ontable d) (ontable e)
 (on b g) (on g h) (on h k) (on k a) (on a f) (on f i) (on j d) (on c e)
 (handempty))
(:goal (and (on a b)))

)