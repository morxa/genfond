(define (problem blocks-015-3)
    (:domain blocksworld)
    (:requirements :strips :typing)
    (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
    (:init (clear b0) (clear b13) (clear b2) (handempty) (on b0 b5) (on b1 b6) (on b10 b8) (on b11 b1) (on b12 b7) (on b14 b4) (on b2 b11) (on b5 b10) (on b6 b9) (on b7 b14) (on b8 b3) (on b9 b12) (ontable b13) (ontable b3) (ontable b4))
    (:goal (and (clear b1) (on b1 b5) (on b5 b4) (on b4 b10) (on b10 b2) (on b2 b7) (on b7 b3) (on b3 b13) (on b13 b14) (ontable b14)))
)