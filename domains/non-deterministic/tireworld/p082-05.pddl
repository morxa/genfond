(define (problem tireworld-082-05)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l9 - location)
    (:init (road l0 l61) (road l0 l71) (road l1 l26) (road l1 l3) (road l10 l12) (road l10 l20) (road l11 l43) (road l11 l47) (road l12 l10) (road l12 l48) (road l13 l33) (road l13 l39) (road l14 l43) (road l14 l65) (road l15 l54) (road l15 l75) (road l16 l67) (road l16 l9) (road l17 l53) (road l17 l79) (road l18 l33) (road l18 l77) (road l19 l59) (road l19 l60) (road l2 l22) (road l2 l25) (road l2 l35) (road l20 l10) (road l20 l34) (road l21 l44) (road l21 l66) (road l22 l2) (road l22 l42) (road l23 l28) (road l23 l41) (road l24 l31) (road l24 l68) (road l25 l2) (road l25 l34) (road l25 l65) (road l26 l1) (road l26 l50) (road l26 l62) (road l27 l29) (road l27 l66) (road l27 l72) (road l28 l23) (road l28 l37) (road l29 l27) (road l29 l7) (road l3 l1) (road l3 l56) (road l30 l62) (road l30 l65) (road l31 l24) (road l31 l58) (road l31 l71) (road l32 l63) (road l32 l66) (road l33 l13) (road l33 l18) (road l34 l20) (road l34 l25) (road l35 l2) (road l35 l74) (road l36 l40) (road l36 l6) (road l37 l28) (road l37 l74) (road l38 l48) (road l38 l70) (road l39 l13) (road l39 l58) (road l4 l64) (road l4 l72) (road l40 l36) (road l40 l79) (road l41 l23) (road l41 l80) (road l42 l22) (road l42 l51) (road l43 l11) (road l43 l14) (road l44 l21) (road l44 l69) (road l45 l76) (road l45 l81) (road l46 l57) (road l46 l59) (road l47 l11) (road l47 l64) (road l48 l12) (road l48 l38) (road l48 l69) (road l49 l69) (road l49 l81) (road l5 l53) (road l5 l55) (road l5 l67) (road l50 l26) (road l50 l9) (road l51 l42) (road l51 l56) (road l52 l73) (road l52 l8) (road l53 l17) (road l53 l5) (road l54 l15) (road l54 l73) (road l55 l5) (road l55 l78) (road l56 l3) (road l56 l51) (road l57 l46) (road l57 l8) (road l58 l31) (road l58 l39) (road l59 l19) (road l59 l46) (road l6 l36) (road l6 l70) (road l60 l19) (road l60 l80) (road l61 l0) (road l61 l75) (road l62 l26) (road l62 l30) (road l63 l32) (road l63 l70) (road l64 l4) (road l64 l47) (road l65 l14) (road l65 l25) (road l65 l30) (road l66 l21) (road l66 l27) (road l66 l32) (road l67 l16) (road l67 l5) (road l68 l24) (road l68 l76) (road l69 l44) (road l69 l48) (road l69 l49) (road l69 l78) (road l7 l29) (road l70 l38) (road l70 l6) (road l70 l63) (road l71 l0) (road l71 l31) (road l72 l27) (road l72 l4) (road l73 l52) (road l73 l54) (road l74 l35) (road l74 l37) (road l75 l15) (road l75 l61) (road l76 l45) (road l76 l68) (road l77 l18) (road l78 l55) (road l78 l69) (road l79 l17) (road l79 l40) (road l8 l52) (road l8 l57) (road l80 l41) (road l80 l60) (road l81 l45) (road l81 l49) (road l9 l16) (road l9 l50) (spare-in l0) (spare-in l1) (spare-in l11) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (vehicle-at l7))
    (:goal (vehicle-at l77))
)