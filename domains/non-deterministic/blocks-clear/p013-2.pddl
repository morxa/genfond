(define (problem blocks-013-2)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b12) (clear b2) (clear b5) (clear b7) (handempty) (on b0 b3) (on b10 b1) (on b11 b6) (on b2 b8) (on b3 b10) (on b4 b9) (on b8 b4) (on b9 b11) (ontable b1) (ontable b12) (ontable b5) (ontable b6) (ontable b7))
    (:goal (clear b4))
)